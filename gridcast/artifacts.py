"""Compact, non-pickle fitted models for scheduled training and hourly inference."""
import hashlib,json,time,warnings
from pathlib import Path
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor,Booster
from statsmodels.tsa.statespace.sarimax import SARIMAX
from .models import features,seasonal_baseline
from .tuning import design,CANDIDATES
from .data import REGIONS,STEP

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def sarima_state(fit,order,seasonal_order,offsets):
 # Low-memory fitting retains the final predicted state in filter_results,
 # even though the high-level result deliberately hides the full state history.
 result=fit.filter_results
 return {'order':list(order),'seasonal_order':list(seasonal_order),'params':fit.params.tolist(),
 'state':result.predicted_state[...,-1].tolist(),'covariance':result.predicted_state_cov[...,-1].tolist(),
 'offsets':np.asarray(offsets).tolist(),'converged':bool(fit.mle_retvals.get('converged',False))}

def sarima_predict(state,new_hourly,hours):
 # Missing endog over the entire forecast horizon prevents observation updates.
 y=np.r_[np.asarray(new_hourly,dtype=float),np.full(hours,np.nan)]
 model=SARIMAX(y,order=state['order'],seasonal_order=state['seasonal_order'],trend='n',simple_differencing=False,enforce_stationarity=True,enforce_invertibility=True)
 model.ssm.initialize_known(np.asarray(state['state']),np.asarray(state['covariance']))
 result=model.filter(np.asarray(state['params']),cov_type='none')
 pred=np.asarray(result.filter_results.forecasts[0,-hours:])
 return np.repeat(pred,4)+np.tile(state['offsets'],hours)

def train_bundle(data,selection,cutoff,directory):
 cutoff=pd.Timestamp(cutoff);directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
 if cutoff!=cutoff.floor('h'):raise ValueError('Model cutoff must be a full hour')
 train=data.loc[data.index<cutoff,list(REGIONS)]
 if not train.index.equals(pd.date_range(train.index[0],cutoff,freq=STEP,inclusive='left')) or not np.isfinite(train.to_numpy()).all():raise ValueError('Complete training data required')
 spec={'schema':'gridcast-compact-1','cutoff':cutoff.isoformat(),'train_start':train.index[0].isoformat(),
 'training_rows':len(train),'data_sha256':hashlib.sha256(train.to_csv().encode()).hexdigest(),
 'selection':selection,'regions':{},'models':['seasonal_naive','lightgbm','sarima']}
 for region in REGIONS:
  tick=time.perf_counter();s=train[region];x,y=design(s.to_numpy(),s.index)
  frozen=selection.get('candidate_specs',CANDIDATES)
  name=selection['winners'][region]['lightgbm'];params=frozen[name]['parameters'];linear=params.get('linear_tree',False)
  mean=x.mean(0) if linear else np.zeros(x.shape[1]);scale=x.std(0) if linear else np.ones(x.shape[1]);scale[scale<1e-8]=1
  gb=LGBMRegressor(**params,random_state=42,n_jobs=1,verbosity=-1,deterministic=True,force_col_wise=True,objective='regression').fit((x-mean)/scale,y)
  gbpath=directory/f'{region}.lightgbm.txt';gb.booster_.save_model(str(gbpath));gb_seconds=time.perf_counter()-tick;tick=time.perf_counter()
  name_s=selection['winners'][region]['sarima'];sp=frozen[name_s]['parameters']
  hourly=s.resample('1h',origin=cutoff).mean().to_numpy();shaped=s.iloc[-28*96:].to_numpy().reshape(-1,4);offsets=(shaped-shaped.mean(1,keepdims=True)).mean(0)
  with warnings.catch_warnings(record=True) as caught:
   fit=SARIMAX(hourly,order=sp['order'],seasonal_order=sp['seasonal_order'],trend='n',simple_differencing=False,enforce_stationarity=True,enforce_invertibility=True).fit(disp=False,maxiter=sp['maxiter'],cov_type='none',low_memory=True)
  state=sarima_state(fit,sp['order'],sp['seasonal_order'],offsets)
  if not state['converged']:raise ValueError(f'SARIMA fit failed convergence for {region}; no model promotion')
  entry={'lightgbm':{'candidate':name,'file':gbpath.name,'sha256':digest(gbpath),'bytes':gbpath.stat().st_size,'mean':mean.tolist(),'scale':scale.tolist(),'fit_seconds':gb_seconds},
   'sarima':{**state,'candidate':name_s,'fit_seconds':time.perf_counter()-tick,'warnings':[str(w.message) for w in caught]}}
  spec['regions'][region]=entry
 body=json.dumps(spec,sort_keys=True,allow_nan=False).encode();spec['model_id']=hashlib.sha256(body).hexdigest()[:24]
 train.to_parquet(directory/'training_reference.parquet',compression='zstd')
 spec['training_reference_sha256']=digest(directory/'training_reference.parquet')
 (directory/'manifest.json').write_text(json.dumps(spec,indent=2,allow_nan=False),encoding='utf-8')
 return spec

class ModelBundle:
 def __init__(self,directory):
  self.root=Path(directory);self.meta=json.loads((self.root/'manifest.json').read_text(encoding='utf-8'));self.models={}
  if self.meta['schema']!='gridcast-compact-1':raise ValueError('Unsupported artifact format')
  for region,info in self.meta['regions'].items():
   p=(self.root/info['lightgbm']['file']).resolve()
   if p.parent!=self.root.resolve() or digest(p)!=info['lightgbm']['sha256']:raise ValueError('Model integrity check failed')
   self.models[region]=Booster(model_file=str(p))
 def predict(self,data,origin,horizon=96,max_age_days=35):
  origin=pd.Timestamp(origin);cutoff=pd.Timestamp(self.meta['cutoff'])
  if origin!=origin.floor('h') or origin<cutoff:raise ValueError('Origin must be an hourly boundary at/after model cutoff')
  if origin-cutoff>pd.Timedelta(days=max_age_days):raise ValueError('Model stale; retraining required')
  targets=pd.date_range(origin,periods=horizon,freq=STEP);outputs={};timing={}
  for region in REGIONS:
   tick=time.perf_counter();s=data.loc[data.index<origin,region]
   required=pd.date_range(origin-pd.Timedelta(days=7),origin,freq=STEP,inclusive='left')
   recent=s.reindex(required)
   if not np.isfinite(recent.to_numpy()).all():raise ValueError('Missing recent observations; forecast withheld')
   info=self.meta['regions'][region];gb=info['lightgbm'];history=list(recent.to_numpy());pred=[]
   for t in targets:
    value=float(self.models[region].predict(((np.asarray(features(history,t))-gb['mean'])/gb['scale']).reshape(1,-1),num_threads=1)[0]);pred.append(value);history.append(value)
   since=s.reindex(pd.date_range(cutoff,origin,freq=STEP,inclusive='left')) if origin>cutoff else s.iloc[:0]
   if not np.isfinite(since.to_numpy()).all():raise ValueError('Missing observations since SARIMA state cutoff')
   hourly=since.to_numpy().reshape(-1,4).mean(1) if len(since) else []
   outputs[region]={'lightgbm':np.asarray(pred),'sarima':sarima_predict(info['sarima'],hourly,int(np.ceil(horizon/4)))[:horizon],
    'seasonal_naive':seasonal_baseline(s,targets)[0]}
   if not all(np.isfinite(v).all() for v in outputs[region].values()):raise ValueError('Nonfinite saved-model prediction')
   timing[region]=time.perf_counter()-tick
  outputs['NEM']={m:sum(outputs[r][m] for r in REGIONS) for m in self.meta['models']}
  return targets,outputs,timing
