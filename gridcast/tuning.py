"""Private, reproducible monthly expanding-window tuning. No production promotion.

Run stages in order: prepare, validate, select, test, curves, report.
Validation fits once/month, issues a 24h forecast at each market midnight.
Past actuals update lag inputs / SARIMA filtered state, never fitted parameters.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[_key]='1'
import argparse,hashlib,json,time,warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from .data import REGIONS,MARKET_TZ
from .models import features,seasonal_baseline
from .aemo import read_monthly

START='2023-07-01'; VALID_START='2025-08'; VALID_END='2026-07'; TEST='2026-08'
CURRENT_GB=dict(n_estimators=180,learning_rate=.04,num_leaves=15,max_depth=5,min_child_samples=40,reg_lambda=5)
GB_SHAPES=[CURRENT_GB,
 dict(n_estimators=300,learning_rate=.03,num_leaves=15,max_depth=5,min_child_samples=80,reg_lambda=10),
 dict(n_estimators=250,learning_rate=.04,num_leaves=31,max_depth=6,min_child_samples=80,reg_lambda=10),
 dict(n_estimators=120,learning_rate=.05,num_leaves=7,max_depth=3,min_child_samples=120,reg_lambda=20)]
CANDIDATES={}
for i,shape in enumerate(GB_SHAPES):
 for linear in (False,True):
  name='lgb_current' if i==0 and not linear else f'lgb_{i}_linear{int(linear)}'
  CANDIDATES[name]={'family':'lightgbm','parameters':{**shape,'linear_tree':linear,**({'linear_lambda':1 if i<2 else 10} if linear else {})}}
for name,order,seasonal in [('sarima_current',(1,0,0),(1,1,0,24)),('sarima_sma',(1,0,0),(0,1,1,24)),('sarima_arma',(1,0,1),(1,1,0,24)),('sarima_ar2',(2,0,0),(1,1,0,24))]:
 CANDIDATES[name]={'family':'sarima','parameters':{'order':order,'seasonal_order':seasonal,'maxiter':60}}

def boundary(month):return pd.Timestamp(str(month)+'-01',tz=MARKET_TZ).tz_convert('UTC')
def folds():return [(str(m),boundary(m),boundary(m+1)) for m in pd.period_range(VALID_START,VALID_END,freq='M')]
def protocol():
 return {'version':'monthly-tuning-1','start':START,'validation_months':[x[0] for x in folds()],
 'test_month':TEST,'training':'expanding, fixed start; refit at each month boundary',
 'forecast':'daily market midnight, 96 quarter-hours; recursive LightGBM, filtered SARIMA state',
 'selection':'per region/family minimum equal-month mean MAE divided by that month seasonal MAE; all 12 successful folds required; SARIMA must converge in every fold',
 'weather':'excluded: exact archived weather runs do not cover this multi-year experiment',
 'test_policy':'parameters locked before test predictions; no search/early stopping on test',
 'learning_curve_days':[90,180,365,730,'all'],'candidates':CANDIDATES,
 'limitations':'final-vintage demand, not historical as-of ingestion; selected CV scores are optimistic; monthly size chart also changes season'}
def read_json(p):return json.loads(Path(p).read_text())
def write_json(p,value):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,default=str));tmp.replace(p)
def prepare(root):
 root=Path(root);p=root/'protocol.json';expected=json.loads(json.dumps(protocol()))
 if p.exists() and read_json(p)!=expected:raise ValueError('Protocol changed; use a new output directory')
 write_json(p,expected)
 manifest=read_json(root/'source-manifest.json')
 for item in manifest:
  assert hashlib.sha256((root/'raw'/item['file']).read_bytes()).hexdigest()==item['sha256']
 raw=read_monthly(sorted((root/'raw').glob('*.csv')))
 raw['target']=(raw.end-pd.Timedelta(minutes=5)).dt.floor('15min')
 grouped=raw.groupby(['region','target']).value.agg(['mean','count'])
 if not (grouped['count']==3).all():raise ValueError('Incomplete source intervals')
 frame=grouped['mean'].unstack(0).sort_index()[list(REGIONS)]
 expected_index=pd.date_range(boundary('2023-07'),boundary('2026-09'),freq='15min',inclusive='left')
 if not frame.index.equals(expected_index) or not np.isfinite(frame.to_numpy()).all():raise ValueError('Incomplete aligned history')
 frame.to_parquet(root/'demand.parquet',compression='zstd')
 write_json(root/'data-audit.json',{'start':str(frame.index[0]),'end_exclusive':str(frame.index[-1]+pd.Timedelta(minutes=15)),
 'months':38,'quarter_hours_per_region':len(frame),'raw_rows':len(raw),'regions':list(REGIONS),
 'source_files':len(manifest),'sha256':hashlib.sha256((root/'demand.parquet').read_bytes()).hexdigest()})
 print('Prepared',len(frame),'quarter-hours per region',flush=True)

def design(values,index):
 """Vectorized equivalent of production features, strictly lagged."""
 v=pd.Series(np.asarray(values),index=index);local=index.tz_convert(MARKET_TZ);hour=local.hour+local.minute/60
 x=np.column_stack([v.shift(1),v.shift(4),v.shift(96),v.shift(672),v.shift(1).rolling(4).mean(),v.shift(1).rolling(96).mean(),np.sin(2*np.pi*hour/24),np.cos(2*np.pi*hour/24),*[np.asarray(local.dayofweek==d,dtype=float) for d in range(7)]])
 return x[672:],np.asarray(values)[672:]

def recursive_month(model,mean,scale,history,origins):
 """Batch independent daily paths; each starts with only its own prior actuals."""
 paths=np.empty((len(origins),672+96))
 for row,origin in enumerate(origins):paths[row,:672]=history.loc[origin-pd.Timedelta(days=7):origin-pd.Timedelta(minutes=15)].to_numpy()
 for h in range(96):
  i=672+h;times=origins+pd.Timedelta(minutes=15*h);local=times.tz_convert(MARKET_TZ);hour=local.hour+local.minute/60
  x=np.column_stack([paths[:,i-1],paths[:,i-4],paths[:,i-96],paths[:,i-672],paths[:,i-4:i].mean(axis=1),paths[:,i-96:i].mean(axis=1),np.sin(2*np.pi*hour/24),np.cos(2*np.pi*hour/24),*[np.asarray(local.dayofweek==d,dtype=float) for d in range(7)]])
  paths[:,i]=model.booster_.predict((x-mean)/scale,num_threads=1)
  if not np.isfinite(paths[:,i]).all():raise ValueError('Nonfinite recursive forecast')
 return paths[:,672:]

def sarima_month(fit,month_hourly,offsets):
 """Filter observed month with fixed parameters. Dynamic paths use only earlier states.
 get_prediction(dynamic=True) disables observation updates at/after each issue.
 Tests also compare against explicit prefix-only extend/forecast.
 """
 filtered=fit.extend(np.asarray(month_hourly))
 return np.array([np.repeat(np.asarray(filtered.get_prediction(start=i,end=i+23,dynamic=True).predicted_mean),4)+np.tile(offsets,24) for i in range(0,len(month_hourly),24)])

def fit_month(series,start,end,candidate,training_days=None):
 train=series.loc[:start-pd.Timedelta(minutes=15)]
 if training_days is not None:train=train.iloc[-int(training_days)*96:]
 assert train.index[-1]<start and len(train)>=29*96
 origins=pd.date_range(start,end,freq='D',inclusive='left');candidate_spec=CANDIDATES[candidate]
 prep_start=time.perf_counter();values=train.to_numpy();converged=True;notes=[]
 if candidate_spec['family']=='lightgbm':
  x,y=design(values,train.index)
  linear=candidate_spec['parameters'].get('linear_tree',False)
  mean=x.mean(axis=0) if linear else np.zeros(x.shape[1]);scale=x.std(axis=0) if linear else np.ones(x.shape[1]);scale[scale<1e-8]=1
  prep=time.perf_counter()-prep_start;tick=time.perf_counter()
  model=LGBMRegressor(**candidate_spec['parameters'],random_state=42,n_jobs=1,verbosity=-1,deterministic=True,force_col_wise=True,objective='regression')
  model.fit((x-mean)/scale,y);fit_s=time.perf_counter()-tick;tick=time.perf_counter()
  pred=recursive_month(model,mean,scale,series,origins)
 else:
  hourly=train.resample('1h',origin=start).mean().to_numpy()
  shaped=values[-28*96:].reshape(-1,4);offsets=(shaped-shaped.mean(axis=1,keepdims=True)).mean(axis=0)
  prep=time.perf_counter()-prep_start;tick=time.perf_counter();par=candidate_spec['parameters']
  with warnings.catch_warnings(record=True) as caught:
   fit=SARIMAX(hourly,order=par['order'],seasonal_order=par['seasonal_order'],trend='n',simple_differencing=False,enforce_stationarity=True,enforce_invertibility=True).fit(disp=False,maxiter=par['maxiter'],cov_type='none',low_memory=False)
  converged=bool(fit.mle_retvals.get('converged',False));notes=[str(w.message) for w in caught]
  fit_s=time.perf_counter()-tick;tick=time.perf_counter()
  future=series.loc[start:end-pd.Timedelta(minutes=15)].resample('1h',origin=start).mean().to_numpy()
  pred=sarima_month(fit,future,offsets)
 predict_s=time.perf_counter()-tick
 actual=series.loc[start:end-pd.Timedelta(minutes=15)].to_numpy().reshape(-1,96)
 if pred.shape!=actual.shape or not np.isfinite(pred).all():raise ValueError('Invalid prediction shape/value')
 return pred,actual,{'candidate':candidate,'family':candidate_spec['family'],'parameters':candidate_spec['parameters'],
 'training_start':str(train.index[0]),'training_end_exclusive':str(start),'training_rows':len(train),'training_days':len(train)/96,
 'feature_seconds':prep,'fit_seconds':fit_s,'predict_seconds':predict_s,'converged':converged,'warnings':notes,
 'mae_mw':float(np.mean(abs(pred-actual))),'rmse_mw':float(np.sqrt(np.mean((pred-actual)**2)))}

def job(root,stage,month,region,candidate,days=None):
 root=Path(root);key=f'{month}_{region}_{candidate}'+(f'_d{days}' if days else '')
 dest=root/stage;dest.mkdir(exist_ok=True);p=dest/(key+'.npz');meta=dest/(key+'.json')
 if p.exists() and meta.exists():return key+' cached'
 data=pd.read_parquet(root/'demand.parquet')[region];start=boundary(month);end=boundary(pd.Period(month)+1)
 tick=time.perf_counter()
 if candidate=='seasonal_naive':
  pred=np.array([seasonal_baseline(data.loc[:o-pd.Timedelta(minutes=15)],pd.date_range(o,periods=96,freq='15min'))[0] for o in pd.date_range(start,end,freq='D',inclusive='left')])
  actual=data.loc[start:end-pd.Timedelta(minutes=15)].to_numpy().reshape(-1,96)
  info={'candidate':candidate,'family':'baseline','training_days':None,'fit_seconds':0,'feature_seconds':0,'predict_seconds':time.perf_counter()-tick,'converged':True,'mae_mw':float(abs(pred-actual).mean()),'rmse_mw':float(np.sqrt(np.mean((pred-actual)**2)))}
 else:pred,actual,info=fit_month(data,start,end,candidate,days)
 info.update(month=month,region=region,elapsed_seconds=time.perf_counter()-tick,stage=stage)
 np.savez_compressed(p,prediction=pred,actual=actual);write_json(meta,info);return key

def run_tasks(tasks,workers):
 with ThreadPoolExecutor(max_workers=workers) as pool:
  futures=[pool.submit(job,*task) for task in tasks]
  for i,f in enumerate(as_completed(futures),1):print(f'{i}/{len(tasks)} {f.result()}',flush=True)

def select(root):
 rows=[];winners={}
 for region in REGIONS:
  winners[region]={}
  for name,spec in CANDIDATES.items():
   ratios=[];eligible=True;times=[]
   for month,_,_ in folds():
    info=read_json(root/'validation'/f'{month}_{region}_{name}.json');base=read_json(root/'validation'/f'{month}_{region}_seasonal_naive.json')
    ratios.append(info['mae_mw']/base['mae_mw']);times.append(info['fit_seconds']);eligible=eligible and info['converged']
   rows.append({'region':region,'candidate':name,'family':spec['family'],'mean_monthly_error_ratio':float(np.mean(ratios)),'mean_monthly_fav_pct':float(100*(1-np.mean(ratios))),'mean_fit_seconds':float(np.mean(times)),'median_fit_seconds':float(np.median(times)),'max_fit_seconds':float(np.max(times)),'fits_over_600_seconds':sum(t>600 for t in times),'eligible':eligible})
  for family in ('lightgbm','sarima'):
   choices=[r for r in rows if r['region']==region and r['family']==family and r['eligible']]
   if not choices:raise ValueError(f'No converged {family} candidate for {region}; no test selection allowed')
   winners[region][family]=min(choices,key=lambda x:(x['mean_monthly_error_ratio'],x['candidate']))['candidate']
 payload={'protocol_sha256':hashlib.sha256((root/'protocol.json').read_bytes()).hexdigest(),'winners':winners,'criterion':protocol()['selection'],
          'candidate_specs':{name:CANDIDATES[name] for name in sorted({c for w in winners.values() for c in w.values()})}}
 payload=json.loads(json.dumps(payload))
 if (root/'selection.json').exists():assert read_json(root/'selection.json')==payload,'Selection cannot change after locking'
 pd.DataFrame(rows).to_csv(root/'search-results.csv',index=False)
 write_json(root/'selection.json',payload);print(json.dumps(winners,indent=2))

def main():
 parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','validate','select','test','curves']);parser.add_argument('--output',required=True);parser.add_argument('--workers',type=int,default=2);args=parser.parse_args();root=Path(args.output)
 if args.stage=='prepare':return prepare(root)
 assert read_json(root/'protocol.json')==json.loads(json.dumps(protocol())),'Protocol mismatch'
 if args.stage=='select':return select(root)
 if args.stage=='validate':tasks=[(root,'validation',month,r,c) for month,_,_ in folds() for r in REGIONS for c in ['seasonal_naive',*CANDIDATES]]
 else:
  selection=read_json(root/'selection.json');assert selection['protocol_sha256']==hashlib.sha256((root/'protocol.json').read_bytes()).hexdigest()
  tasks=[]
  for r in REGIONS:
   names=list(dict.fromkeys(['seasonal_naive','lgb_current','sarima_current',*selection['winners'][r].values()]))
   if args.stage=='test':tasks.extend((root,'test',TEST,r,c) for c in names)
   else:tasks.extend((root,'curves',TEST,r,c,d) for c in names if c!='seasonal_naive' for d in [90,180,365,730])
 tick=time.perf_counter();run_tasks(tasks,args.workers);write_json(root/f'{args.stage}-runtime.json',{'wall_seconds':time.perf_counter()-tick,'jobs':len(tasks),'workers':args.workers})
if __name__=='__main__':main()
