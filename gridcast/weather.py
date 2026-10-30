"""Archived GFS weather runs: explicit initialization, conservative availability and provenance.
Capital-city proxies are an initial experiment, not population-weighted regional weather.
"""
import hashlib,json,time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen
import numpy as np
import pandas as pd
from .data import REGIONS
LOCATIONS={'NSW1':(-33.87,151.21),'QLD1':(-27.47,153.03),'VIC1':(-37.81,144.96),'SA1':(-34.93,138.60),'TAS1':(-42.88,147.33)}
VARIABLES=['temperature_2m','relative_humidity_2m','shortwave_radiation']
FEATURES=VARIABLES+['heating_degrees_18','cooling_degrees_22']
ENDPOINT='https://single-runs-api.open-meteo.com/v1/forecast'
class WeatherArchive:
 def __init__(self,root):self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True);self.frames={}
 def path(self,run):return self.root/(pd.Timestamp(run).strftime('%Y%m%dT%H%M')+'.json')
 def fetch(self,run):
  run=pd.Timestamp(run);path=self.path(run)
  if path.exists():return path
  if run.hour!=0:raise ValueError('This experiment uses 00 UTC runs only')
  params={'latitude':','.join(str(LOCATIONS[r][0]) for r in REGIONS),'longitude':','.join(str(LOCATIONS[r][1]) for r in REGIONS),'hourly':','.join(VARIABLES),'models':'gfs_global','run':run.strftime('%Y-%m-%dT%H:%M'),'forecast_days':3,'timezone':'UTC'}
  url=ENDPOINT+'?'+urlencode(params)
  with urlopen(url,timeout=60) as response:raw=response.read()
  data=json.loads(raw)
  if not isinstance(data,list) or len(data)!=5:raise ValueError('Expected five location responses')
  for item in data:
   if 'hourly' not in item:raise ValueError('Missing hourly archive')
  path.write_bytes(raw)
  path.with_suffix('.provenance.json').write_text(json.dumps({'url':url,'retrieved_at':pd.Timestamp.now(tz='UTC').isoformat(),'run_initialized_at':run.isoformat(),'assumed_available_at':(run+pd.Timedelta(hours=8)).isoformat(),'availability_note':'Conservative 8-hour delay; historical actual publication timestamps are not provided.','provider':'Open-Meteo / NOAA GFS','model':'gfs_global','locations':LOCATIONS,'sha256':hashlib.sha256(raw).hexdigest()},indent=2))
  return path
 def frame(self,run,region):
  run=pd.Timestamp(run);key=(run,region)
  if key not in self.frames:
   path=self.path(run)
   if not path.exists():raise ValueError(f'Missing archived weather run {run}; collect first')
   data=json.loads(path.read_text())[list(REGIONS).index(region)]['hourly']
   frame=pd.DataFrame({v:data[v] for v in VARIABLES},index=pd.to_datetime(data['time'],utc=True))
   self.frames[key]=frame
  return self.frames[key]
 def features(self,region,train_index,targets,origin):
  origin=pd.Timestamp(origin)
  if region not in REGIONS:raise ValueError('Regional weather features require a region')
  # Forecasts issued at 14 UTC in the benchmark use the same day 00 UTC run.
  future_run=(origin-pd.Timedelta(hours=8)).floor('D')
  if future_run+pd.Timedelta(hours=8)>origin:raise ValueError('Weather not available by issue time')
  index=train_index.append(targets)
  runs=pd.Series([(t-pd.Timedelta(hours=14)).floor('D') if t<origin else future_run for t in index],index=index)
  result=pd.DataFrame(index=index,columns=VARIABLES,dtype=float);fallbacks=[]
  for run,group in runs.groupby(runs):
   if run+pd.Timedelta(hours=8)>min(group.index.min(),origin):raise ValueError('Future weather run leakage')
   block=self.frame(run,region).reindex(group.index.floor('h')).to_numpy(dtype=float)
   if not np.isfinite(block).all():
    older=run-pd.Timedelta(days=1)
    block=self.frame(older,region).reindex(group.index.floor('h')).to_numpy(dtype=float)
    fallbacks.append({'requested_run':run.isoformat(),'used_run':older.isoformat(),'rows':len(group)})
   result.loc[group.index,VARIABLES]=block
  result['heating_degrees_18']=(18-result.temperature_2m).clip(lower=0)
  result['cooling_degrees_22']=(result.temperature_2m-22).clip(lower=0)
  if not np.isfinite(result.to_numpy()).all():raise ValueError('Weather archive has missing/nonfinite values; no silent fallback')
  result.attrs['provenance']={'older_run_fallbacks':fallbacks,'provider':'Open-Meteo / NOAA GFS','future_run_initialized_at':future_run.isoformat(),'assumed_publication_delay_hours':8,'region_proxy':LOCATIONS[region],'training_forecasts':'00 UTC run for each 14 UTC-to-14 UTC demand day','hourly_alignment':'Previous hourly forecast held constant within quarter-hours'}
  return result

if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser(description='Cache archived daily 00 UTC GFS runs for the five regional proxies')
 parser.add_argument('--cache',required=True);parser.add_argument('--start',required=True);parser.add_argument('--end',required=True)
 args=parser.parse_args();archive=WeatherArchive(args.cache)
 for run in pd.date_range(args.start,args.end,freq='D',tz='UTC'):
  print(archive.fetch(run),flush=True);time.sleep(.35)
