"""Hourly saved-model batch for R2 + GitHub Actions. No cloud resources are created.

Single writer required: hourly and training workflows share one concurrency group.
Forecast vintages and evaluation deltas are immutable; state snapshots are replaceable.
"""
import argparse,io,json,tempfile,time,hashlib
from pathlib import Path
from contextlib import ExitStack
import numpy as np
import pandas as pd
from .data import REGIONS,NAMES,STEP,utc,stamp
from .aemo import DispatchSource,aggregate
from .artifacts import ModelBundle,train_bundle,digest
from .object_store import LocalObjects,R2Objects
from .metrics import leaderboard,forecast_added_value,performance_history,metrics,drift
from .monitoring import compare

MODELS=('seasonal_naive','lightgbm','sarima');SERIES=(*REGIONS,'NEM')
def encode(value):return json.dumps(value,allow_nan=False,default=str,separators=(',',':')).encode()
def parquet(frame):
 b=io.BytesIO();frame.to_parquet(b,index=False,compression='zstd');return b.getvalue()
def load_frame(objects,key):return pd.read_parquet(io.BytesIO(objects.get(key)))
def maybe_frame(objects,key):
 try:return load_frame(objects,key)
 except FileNotFoundError:return pd.DataFrame()
def wide(actuals):return actuals.pivot(index='target',columns='region',values='value').sort_index()[list(REGIONS)]

def merge_actuals(old,incoming,received):
 rows=[]
 for region,series in incoming.items():
  if region=='NEM':continue
  rows.extend({'region':region,'target':t,'value':float(v),'seen_at':received} for t,v in series.items())
 new=pd.DataFrame(rows)
 if old.empty:return new.sort_values(['region','target'])
 indexed=old.set_index(['region','target']);incoming=new.set_index(['region','target'])
 changed=~incoming.index.isin(indexed.index)
 common=incoming.index.intersection(indexed.index)
 changed=changed | incoming.index.isin(common[indexed.loc[common,'value'].to_numpy()!=incoming.loc[common,'value'].to_numpy()])
 updates=incoming.loc[changed].reset_index()
 return pd.concat([old,updates],ignore_index=True).drop_duplicates(['region','target'],keep='last').sort_values(['region','target'])

def evaluated(forecasts,actuals):
 if forecasts.empty:return pd.DataFrame()
 a=actuals.copy();w=wide(a);valid=w.dropna();seen=a.groupby('target').seen_at.max().reindex(valid.index)
 nem=pd.DataFrame({'region':'NEM','target':valid.index,'value':valid.sum(axis=1).to_numpy(),'seen_at':seen.to_numpy()})
 a=pd.concat([a,nem],ignore_index=True).rename(columns={'value':'actual','seen_at':'actual_seen_at'})
 scored=forecasts.merge(a,on=['region','target'],how='inner',validate='many_to_one');scored['error']=scored.prediction-scored.actual
 return scored

def interval_widths(scores,cutoff):
 cutoff=pd.Timestamp(cutoff);s=scores[(scores.target<cutoff)&(scores.target>=cutoff-pd.Timedelta(days=30))]
 widths={}
 for r in SERIES:
  widths[r]={}
  for m in MODELS:
   widths[r][m]=[]
   for b in range(4):
    e=abs(s[(s.region==r)&(s.model==m)&(s.horizon>b*24)&(s.horizon<=(b+1)*24)].error.to_numpy())
    if len(e)<48:raise ValueError('At least 48 past errors per model/region/lead bucket required for intervals')
    widths[r][m].append(float(np.sort(e)[min(len(e)-1,int(np.ceil((len(e)+1)*.9))-1)]))
 return {'widths':widths,'calibration_end_exclusive':cutoff.isoformat(),'method':'90% rolling past forecast-error quantile; deployment-model residuals, not guaranteed coverage'}

def load_models(objects,directory):
 pointer=json.loads(objects.get('models/current.json'));prefix=pointer['prefix']
 if not prefix.startswith('models/') or '..' in prefix:raise ValueError('Invalid model pointer')
 directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
 body=objects.get(prefix+'/manifest.json')
 if hashlib.sha256(body).hexdigest()!=pointer['manifest_sha256']:raise ValueError('Manifest checksum mismatch')
 manifest=json.loads(body)
 names=['manifest.json','training_reference.parquet','calibration.json']+[x['lightgbm']['file'] for x in manifest['regions'].values()]
 for name in names:
  if Path(name).name!=name:raise ValueError('Invalid model file')
  (directory/name).write_bytes(body if name=='manifest.json' else objects.get(prefix+'/'+name))
 if digest(directory/'training_reference.parquet')!=manifest['training_reference_sha256']:raise ValueError('Reference checksum mismatch')
 return ModelBundle(directory),json.loads((directory/'calibration.json').read_text())

def publish_models(objects,directory):
 directory=Path(directory);m=json.loads((directory/'manifest.json').read_text());prefix='models/'+m['model_id']
 for p in directory.iterdir():
  if p.is_file():objects.put(prefix+'/'+p.name,p.read_bytes(),immutable=True)
 objects.put('models/current.json',encode({'prefix':prefix,'manifest_sha256':digest(directory/'manifest.json'),'model_id':m['model_id'],'cutoff':m['cutoff']}))

def collect_recent_forecasts(objects,now,known):
 frames=[known] if len(known) else [];ids=set(known.run_id) if len(known) else set()
 for day in pd.date_range(now.floor('D')-pd.Timedelta(days=7),now.floor('D'),freq='D'):
  for key in objects.keys('forecasts/'+day.strftime('%Y-%m-%d')+'/'):
   run_id=Path(key).stem
   if run_id not in ids:frames.append(load_frame(objects,key));ids.add(run_id)
 if not frames:return pd.DataFrame()
 combined=pd.concat(frames,ignore_index=True).drop_duplicates(['run_id','region','model','target'])
 return combined.loc[combined.issued_at>=now-pd.Timedelta(days=7)]

def public_payloads(actuals,forecasts,scores,bundle,received,reference,monitoring_cache=None):
 result={};w=wide(actuals);w['NEM']=w.sum(axis=1,min_count=5);ref=reference.copy();ref['NEM']=ref.sum(axis=1,min_count=5)
 for r in SERIES:
  f=forecasts[forecasts.region==r];s=scores[scores.region==r] if len(scores) else pd.DataFrame()
  latest=f.sort_values('issued_at').iloc[-1];run=f[f.run_id==latest.run_id];series=w[r].dropna();origin=latest.origin
  metadata={'model_id':latest.model_id,'train_rows':bundle.meta['training_rows'],'train_start':bundle.meta['train_start'],'nominal_horizon':96,
   'availability':'prospective received observations; saved model inference','negative_values':'preserved','interval_method':'90% past residual quantile by 6-hour bucket','drift':drift(series,origin)}
  horizon=[]
  if len(s):horizon=[{'model':m,'hours':int(b),**metrics(g)} for (m,b),g in s.assign(bucket=((s.horizon-1)//24+1)*6).groupby(['model','bucket'])]
  vintages=[]
  for rid,v in f.sort_values('issued_at').groupby('run_id',sort=False):
   vintages.append({'id':rid,'issued_at':str(v.issued_at.iloc[0]),'train_cutoff':str(v.origin.iloc[0]),'forecasts':v[['model','target','prediction','lower','upper']].assign(target=lambda x:x.target.map(lambda t:t.isoformat())).to_dict('records')})
  current=series.loc[(series.index>=pd.Timestamp(bundle.meta['cutoff']))&(series.index>=received-pd.Timedelta(days=7))]
  # KDE/statistical tests are refreshed every six hours; the API keeps their true current_end.
  monitor=monitoring_cache.get(r) if monitoring_cache else None
  if monitor is None:
   train=ref[r].dropna()
   def slots(index):
    local=index.tz_convert('Etc/GMT-10');return local.dayofweek*96+local.hour*4+local.minute//15
   typical=train.groupby(slots(train.index)).mean()
   comparisons={'demand':compare(train,current),'seasonal_residual':compare(train.to_numpy()-typical.reindex(slots(train.index)).to_numpy(),current.to_numpy()-typical.reindex(slots(current.index)).to_numpy()),'ramp':compare(train.diff(),current.reindex(pd.date_range(current.index[0],current.index[-1],freq=STEP)).diff() if len(current) else [])}
   family=[t for c in comparisons.values() for t in c.get('tests',[]) if t['p_value'] is not None];adjusted=0
   for i,t in enumerate(sorted(family,key=lambda x:x['p_value'])):
    adjusted=max(adjusted,min(1.,(len(family)-i)*t['p_value']));t['holm_p_value']=adjusted
   monitor=[{'run_id':latest.run_id,'cutoff':bundle.meta['cutoff'],'reference_start':bundle.meta['train_start'],'current_end':str(current.index[-1]+STEP) if len(current) else bundle.meta['cutoff'],
    'status':'ready' if len(current)>=96 else 'short_window','training_missing_fraction':0,'current_missing_pct':0,'current_expected_n':len(current),
    'availability':'saved training reference versus subsequently observed demand','comparisons':comparisons}]
  fav=forecast_added_value(s) if len(s) else []
  actual_context=series.loc[origin-pd.Timedelta(days=5):].tail(480)
  result[r]={'schema_version':'1.0','source':'aemo','region':r,'region_name':NAMES[r],'kind':'live','available_modes':['live'],'generated_at':received.isoformat(),'data_updated_at':actuals.seen_at.max().isoformat(),
   'latest_run':{'id':latest.run_id,'issued_at':latest.issued_at.isoformat(),'created_at':latest.issued_at.isoformat(),'train_cutoff':origin.isoformat(),'metadata':metadata},
   'forecasts':run[['model','target','prediction','lower','upper','horizon']].assign(target=lambda x:x.target.map(lambda t:t.isoformat())).to_dict('records'),
   'leaderboard':leaderboard(s) if len(s) else [],'forecast_added_value':fav,'backtest_fav':[],
   'performance_history':performance_history(s) if len(s) else [],'by_horizon':horizon,'monitoring':monitor,
   'forecast_count':len(f),'scored_count':len(s),'pending_count':len(f)-len(s),'events':[],
   'actuals':[{'target':t.isoformat(),'value':float(v)} for t,v in actual_context.items()],
   'evolution':{'vintages':vintages[-96:],'actuals':[{'target':t.isoformat(),'value':float(v)} for t,v in series.tail(9*96).items()],'limit':96,'kind':'live'},
   'data_range':{'start':str(series.index[0]),'end':str(series.index[-1]+STEP)},'reconciliation':'Sum of five regional point forecasts; intervals calibrated on aggregate errors'}
 previews=[{'region':r,'name':NAMES[r],'kind':'live',**{k:result[r][k] for k in ['forecasts','actuals','backtest_fav','forecast_added_value','data_updated_at','latest_run']}} for r in REGIONS]
 for item in result.values():item['regional_previews']=previews
 return result

def cycle(objects,work,source=None,received=None):
 work=Path(work);work.mkdir(parents=True,exist_ok=True);received=utc() if received is None else pd.Timestamp(received)
 old=load_frame(objects,'state/actuals.parquet')
 age=received-(wide(old).dropna().index[-1]+STEP)
 reports=max(24,min(288,int(np.ceil(age.total_seconds()/300))+24))
 source=source or DispatchSource(work/'dispatch-cache',max_reports=reports)
 raw=source.fetch();incoming,quality=aggregate(raw,received);actuals=merge_actuals(old,incoming,received);data=wide(actuals)
 end=data.dropna().index[-1]+STEP
 if received-end>pd.Timedelta(minutes=30) or end>received:raise ValueError('Stale or future observations; no live forecast published')
 bundle,calibration=load_models(objects,work/'models');anchor=end.floor('h')
 if pd.Timestamp(calibration['calibration_end_exclusive'])>anchor:raise ValueError('Future calibration leakage')
 if anchor-pd.Timestamp(calibration['calibration_end_exclusive'])>pd.Timedelta(days=90):raise ValueError('Calibration stale; refresh required')
 rid=received.strftime('%Y%m%dT%H')+'-'+bundle.meta['model_id'];key='forecasts/'+received.strftime('%Y-%m-%d')+'/'+rid+'.parquet'
 try:current=load_frame(objects,key);created=False
 except FileNotFoundError:
  targets,preds,timing=bundle.predict(data,anchor);records=[]
  for r,models in preds.items():
   for m,values in models.items():
    widths=np.repeat(calibration['widths'][r][m],24)
    for i,t in enumerate(targets):
     if t<received.ceil('15min'):continue
     records.append({'run_id':rid,'model_id':bundle.meta['model_id'],'issued_at':received,'origin':anchor,'region':r,'model':m,'target':t,'horizon':i+1,'prediction':float(values[i]),'lower':float(values[i]-widths[i]),'upper':float(values[i]+widths[i])})
  current=pd.DataFrame(records)
  objects.put(key,parquet(current),immutable=True);created=True
 known=maybe_frame(objects,'state/recent-forecasts.parquet');forecasts=collect_recent_forecasts(objects,received,known)
 scores=evaluated(forecasts,actuals);previous=maybe_frame(objects,'state/recent-scores.parquet')
 keys=['run_id','region','model','target','actual_seen_at']
 delta=scores if previous.empty else scores.merge(previous[keys].assign(_old=True),on=keys,how='left').query('_old != True').drop(columns='_old')
 suffix=received.strftime('%Y%m%dT%H%M%S%f')
 if len(delta):objects.put('evaluations/'+received.strftime('%Y-%m-%d')+'/'+suffix+'.parquet',parquet(delta),immutable=True)
 objects.put('observations/'+received.strftime('%Y-%m-%d')+'/'+suffix+'.parquet',parquet(raw.assign(received_at=received)),immutable=True)
 reference=pd.read_parquet(work/'models/training_reference.parquet');monitor=None
 try:
  cached=json.loads(objects.get('state/monitoring.json'))
  if cached['model_id']==bundle.meta['model_id'] and received-pd.Timestamp(cached['computed_at'])<pd.Timedelta(hours=6):monitor=cached['regions']
 except FileNotFoundError:pass
 payloads=public_payloads(actuals,forecasts,scores,bundle,received,reference,monitor)
 objects.put('state/actuals.parquet',parquet(actuals));objects.put('state/recent-forecasts.parquet',parquet(forecasts));objects.put('state/recent-scores.parquet',parquet(scores))
 if monitor is None:objects.put('state/monitoring.json',encode({'model_id':bundle.meta['model_id'],'computed_at':received.isoformat(),'regions':{r:p['monitoring'] for r,p in payloads.items()}}))
 for r,payload in payloads.items():
  objects.put(f'public/snapshots/{suffix}/{r}.json',encode(payload))
  objects.put(f'public/snapshots/{suffix}/{r}.csv',pd.DataFrame(payload['forecasts']).to_csv(index=False).encode())
 objects.put('public/current.json',encode({'snapshot':suffix,'generated_at':received.isoformat(),'forecast_issued_at':current.issued_at.iloc[0].isoformat(),'model_id':bundle.meta['model_id']}))
 # Public JSON/CSV are derived caches. Preserve all immutable forecast/evaluation archives.
 threshold=(received-pd.Timedelta(hours=4)).strftime('%Y%m%dT%H%M%S%f')
 for old_key in objects.keys('public/snapshots/'):
  if old_key.split('/')[2]<threshold:objects.delete_public_cache(old_key)
 return {'forecast_created':created,'run_id':rid,'new_scores':len(delta),'source_end':str(end),'quality':quality}

def main():
 parser=argparse.ArgumentParser();parser.add_argument('command',choices=['hourly','bootstrap','train']);parser.add_argument('--local-store');parser.add_argument('--work',default='.hourly-work');parser.add_argument('--data');parser.add_argument('--models');parser.add_argument('--selection');parser.add_argument('--calibration');args=parser.parse_args()
 objects=LocalObjects(args.local_store) if args.local_store else R2Objects();work=Path(args.work);work.mkdir(parents=True,exist_ok=True)
 if args.command=='hourly':print(json.dumps(cycle(objects,work),indent=2));return
 if args.command=='bootstrap':
  if not args.data or not args.models:parser.error('bootstrap requires --data wide Parquet and --models folder')
  try:objects.get('state/actuals.parquet')
  except FileNotFoundError:pass
  else:raise ValueError('Bootstrap refuses to overwrite existing state')
  data=pd.read_parquet(args.data);data.index=pd.to_datetime(data.index,utc=True)
  if utc()-(data.index[-1]+STEP)>pd.Timedelta(hours=1):raise ValueError('Bootstrap demand is stale; backfill first')
  rows=data[list(REGIONS)].rename_axis('target').reset_index().melt(id_vars='target',var_name='region',value_name='value').assign(seen_at=utc())
  publish_models(objects,args.models);objects.put('state/actuals.parquet',parquet(rows));return
 if not args.selection:parser.error('train requires a locked --selection JSON')
 actuals=load_frame(objects,'state/actuals.parquet');data=wide(actuals);cutoff=(data.dropna().index[-1]+STEP).floor('h')
 if utc()-cutoff>pd.Timedelta(hours=2):raise ValueError('Training observations stale')
 destination=work/'new-models';train_bundle(data,json.loads(Path(args.selection).read_text()),cutoff,destination)
 scores=maybe_frame(objects,'state/recent-scores.parquet')
 if len(scores):cal=interval_widths(scores,cutoff)
 elif args.calibration:cal=json.loads(Path(args.calibration).read_text())
 else:raise ValueError('Bootstrap needs a past-only calibration file')
 if pd.Timestamp(cal['calibration_end_exclusive'])>cutoff:raise ValueError('Future calibration leakage')
 (destination/'calibration.json').write_bytes(encode(cal));publish_models(objects,destination)
if __name__=='__main__':main()
