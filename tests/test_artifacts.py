import unittest,json
import numpy as np
import pandas as pd
from unittest.mock import patch
from statsmodels.tsa.statespace.sarimax import SARIMAX
from gridcast.artifacts import sarima_state,sarima_predict,ModelBundle
from gridcast.data import REGIONS

class ArtifactTests(unittest.TestCase):
 def test_compact_state_roundtrip_matches_forecasts_without_refitting(self):
  rng=np.random.default_rng(4);y=100+10*np.sin(np.arange(650)*2*np.pi/24)+rng.normal(0,1,650)
  fit=SARIMAX(y[:600],order=(1,0,0),seasonal_order=(1,1,0,24),trend='n').fit(disp=False,maxiter=60,cov_type='none',low_memory=True)
  state=json.loads(json.dumps(sarima_state(fit,(1,0,0),(1,1,0,24),np.zeros(4))))
  np.testing.assert_allclose(sarima_predict(state,[],24),np.repeat(fit.forecast(24),4),atol=1e-7)
  reference=SARIMAX(y[:624],order=(1,0,0),seasonal_order=(1,1,0,24),trend='n').filter(fit.params)
  np.testing.assert_allclose(sarima_predict(state,y[600:624],24),np.repeat(reference.forecast(24),4),atol=1e-7)
  self.assertLess(len(json.dumps(state)),100000)
  class Tree:
   def predict(self,x,num_threads):return np.ones(len(x))
  bundle=object.__new__(ModelBundle);cutoff=pd.Timestamp('2026-01-01',tz='UTC')
  bundle.meta={'cutoff':cutoff.isoformat(),'models':['seasonal_naive','lightgbm','sarima'],'regions':{r:{'lightgbm':{'mean':np.zeros(15).tolist(),'scale':np.ones(15).tolist()},'sarima':state} for r in REGIONS}}
  bundle.models={r:Tree() for r in REGIONS}
  data=pd.DataFrame({r:np.ones(7*96) for r in REGIONS},index=pd.date_range(cutoff-pd.Timedelta(days=7),cutoff,freq='15min',inclusive='left'))
  with patch('gridcast.artifacts.seasonal_baseline',return_value=(np.ones(96),{})):
   targets,pred,_=bundle.predict(data,cutoff)
   self.assertEqual(len(targets),96);np.testing.assert_allclose(pred['NEM']['lightgbm'],5)
if __name__=='__main__':unittest.main()
