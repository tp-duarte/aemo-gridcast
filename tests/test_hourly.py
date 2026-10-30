import unittest,tempfile,json
from pathlib import Path
from unittest.mock import patch
import numpy as np,pandas as pd
from gridcast.hourly import cycle,parquet,wide,merge_actuals
from gridcast.object_store import LocalObjects
from gridcast.data import REGIONS

class HourlyTests(unittest.TestCase):
 def test_local_two_hour_cycle_scores_without_overwriting_forecasts(self):
  with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
   root=Path(tmp);objects=LocalObjects(root/'objects');now=pd.Timestamp('2026-01-03T14:17Z')
   index=pd.date_range(now.floor('D')-pd.Timedelta(days=3),now.floor('15min'),freq='15min',inclusive='left')
   data=pd.DataFrame({r:100+i+np.sin(np.arange(len(index))/10) for i,r in enumerate(REGIONS)},index=index)
   actual=data.rename_axis('target').reset_index().melt(id_vars='target',var_name='region',value_name='value').assign(seen_at=now-pd.Timedelta(hours=1))
   objects.put('state/actuals.parquet',parquet(actual));work=root/'work';(work/'models').mkdir(parents=True)
   data.iloc[:200].to_parquet(work/'models/training_reference.parquet')
   class Bundle:
    meta={'model_id':'testmodel','cutoff':str(index[200]),'training_rows':200,'train_start':str(index[0])}
    def predict(self,history,origin):
     targets=pd.date_range(origin,periods=96,freq='15min');p={r:{m:np.full(96,100+i) for m in ['seasonal_naive','lightgbm','sarima']} for i,r in enumerate(REGIONS)}
     p['NEM']={m:sum(p[r][m] for r in REGIONS) for m in ['seasonal_naive','lightgbm','sarima']};return targets,p,{}
   class Source:
    def __init__(self,t):self.t=t
    def fetch(self):
     times=pd.date_range(self.t.floor('15min')-pd.Timedelta(minutes=115),self.t.floor('15min'),freq='5min')
     return pd.DataFrame([{'region':r,'end':t,'value':101+i} for i,r in enumerate(REGIONS) for t in times])
   cal={'calibration_end_exclusive':str(index[200]),'widths':{r:{m:[20]*4 for m in ['seasonal_naive','lightgbm','sarima']} for r in [*REGIONS,'NEM']}}
   with patch('gridcast.hourly.load_models',return_value=(Bundle(),cal)):
    a=cycle(objects,work,Source(now),now);self.assertTrue(a['forecast_created']);self.assertEqual(a['new_scores'],0)
    first=objects.keys('forecasts/')[0];saved=objects.get(first)
    b=cycle(objects,work,Source(now+pd.Timedelta(hours=1)),now+pd.Timedelta(hours=1));self.assertGreater(b['new_scores'],0)
    self.assertEqual(saved,objects.get(first))
    again=cycle(objects,work,Source(now+pd.Timedelta(hours=1)),now+pd.Timedelta(hours=1));self.assertFalse(again['forecast_created']);self.assertEqual(again['new_scores'],0)
    pointer=json.loads(objects.get('public/current.json'));payload=json.loads(objects.get('public/snapshots/'+pointer['snapshot']+'/NEM.json'))
    self.assertEqual(payload['kind'],'live');self.assertGreater(len(payload['forecast_added_value']),0)
    self.assertEqual(payload['backtest_fav'],[])
    with self.assertRaises(ValueError):objects.delete_public_cache(first)
    with self.assertRaises(ValueError):objects.get('../outside')
if __name__=='__main__':unittest.main()
