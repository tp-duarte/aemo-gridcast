import unittest,tempfile,json
from pathlib import Path
from unittest.mock import patch
import numpy as np,pandas as pd
from gridcast.hourly import cycle,parquet,wide,merge_actuals,evaluated,normalize_times,recover_missing_dispatch,load_frame
from gridcast.object_store import LocalObjects
from gridcast.data import REGIONS

class HourlyTests(unittest.TestCase):
 def test_recovery_repairs_internal_hole_before_latest_complete_reading(self):
  times=pd.date_range('2026-11-08T08:00Z',periods=5,freq='15min')
  actual=pd.DataFrame([{'region':r,'target':t,'value':100.,'seen_at':times[-1]} for r in REGIONS for t in times if t!=times[1]])
  raw=pd.DataFrame([{'region':r,'end':times[-1]+pd.Timedelta(minutes=5),'value':101.} for r in REGIONS])
  official=pd.DataFrame([{'region':r,'end':times[1]+pd.Timedelta(minutes=m),'value':100.} for r in REGIONS for m in [5,10,15]])
  with patch('gridcast.hourly.download_months',return_value=[{'file':'fixture.csv'}]),patch('gridcast.hourly.read_monthly',return_value=official):
   result=recover_missing_dispatch(raw,actual,times[0],times[-1],Path('unused-test-cache'))
  self.assertEqual(len(result),len(raw)+15)
  self.assertEqual(set(result[result.source=='AEMO monthly TRADE'].end),set(official.end))
 def test_mixed_arrow_timestamp_precision_preserves_evaluation_identity(self):
  t=pd.Timestamp('2026-10-30T16:00Z');seen=pd.Timestamp('2026-10-30T16:29:44.192331Z')
  actual=pd.DataFrame({'region':list(REGIONS),'target':[t]*5,'value':[100.]*5,'seen_at':[seen]*5})
  actual['seen_at']=actual.seen_at.dt.as_unit('us')
  forecast=pd.DataFrame({'region':[*REGIONS,'NEM'],'target':[t]*6,'prediction':[100.]*5+[500.]})
  scored=evaluated(forecast,actual)
  self.assertEqual(str(scored.actual_seen_at.dtype),'datetime64[ns, UTC]')
  previous=scored.copy();previous['actual_seen_at']=previous.actual_seen_at.dt.as_unit('us')
  previous=normalize_times(previous)
  joined=scored.merge(previous[['region','target','actual_seen_at']],on=['region','target','actual_seen_at'])
  self.assertEqual(len(joined),6)
  self.assertTrue((joined.actual_seen_at==seen).all())
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
    with patch('gridcast.hourly.recover_missing_dispatch',wraps=recover_missing_dispatch) as recovery:
     a=cycle(objects,work,Source(now),now)
     self.assertEqual(recovery.call_args.args[2],index[0])
    self.assertTrue(a['forecast_created']);self.assertEqual(a['new_scores'],0)
    first=objects.keys('forecasts/')[0];saved=objects.get(first)
    b=cycle(objects,work,Source(now+pd.Timedelta(hours=1)),now+pd.Timedelta(hours=1));self.assertGreater(b['new_scores'],0)
    self.assertEqual(saved,objects.get(first))
    again=cycle(objects,work,Source(now+pd.Timedelta(hours=1)),now+pd.Timedelta(hours=1));self.assertFalse(again['forecast_created']);self.assertEqual(again['new_scores'],0)
    later=now+pd.Timedelta(hours=1,minutes=30,microseconds=192331)
    recovered=cycle(objects,work,Source(later),later);self.assertFalse(recovered['forecast_created']);self.assertGreater(recovered['new_scores'],0)
    self.assertEqual(saved,objects.get(first))
    pointer=json.loads(objects.get('public/current.json'));payload=json.loads(objects.get('public/snapshots/'+pointer['snapshot']+'/NEM.json'))
    self.assertEqual(payload['kind'],'live');self.assertGreater(len(payload['forecast_added_value']),0)
    self.assertEqual(payload['latest_run']['train_cutoff'],Bundle.meta['cutoff'])
    self.assertNotEqual(payload['latest_run']['forecast_origin'],Bundle.meta['cutoff'])
    self.assertEqual(payload['backtest_fav'],[])
    with self.assertRaises(ValueError):objects.delete_public_cache(first)
    with self.assertRaises(ValueError):objects.get('../outside')
    before=objects.keys('forecasts/')
    with patch.object(Bundle,'predict',side_effect=ValueError('Missing recent observations')):
     with self.assertRaisesRegex(ValueError,'Missing recent'):
      cycle(objects,work,Source(now+pd.Timedelta(hours=2)),now+pd.Timedelta(hours=2))
    self.assertEqual(before,objects.keys('forecasts/'))
    self.assertGreater(load_frame(objects,'state/actuals.parquet').target.max(),now+pd.Timedelta(hours=1))
if __name__=='__main__':unittest.main()

