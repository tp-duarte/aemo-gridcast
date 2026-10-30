import unittest
import numpy as np
import pandas as pd
import tempfile,json
from pathlib import Path
from unittest.mock import patch
from statsmodels.tsa.statespace.sarimax import SARIMAX
from gridcast.tuning import design,features,recursive_month,sarima_month,folds,boundary,CANDIDATES,select,protocol

class TuningTests(unittest.TestCase):
 def test_selection_uses_only_validation_and_locks_winner(self):
  with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
   root=Path(tmp);(root/'validation').mkdir();(root/'test').mkdir()
   (root/'protocol.json').write_text(json.dumps(protocol()))
   for month,_,_ in folds():
    for name in ['seasonal_naive',*CANDIDATES]:
     score=50 if name=='lgb_0_linear1' else 100
     (root/'validation'/f'{month}_NSW1_{name}.json').write_text(json.dumps({'mae_mw':score,'fit_seconds':1,'converged':True}))
   (root/'test'/'unrelated.json').write_text(json.dumps({'candidate':'lgb_current','mae_mw':0}))
   with patch('gridcast.tuning.REGIONS',('NSW1',)):
    select(root);locked=json.loads((root/'selection.json').read_text());select(root)
    self.assertEqual(locked['winners']['NSW1']['lightgbm'],'lgb_0_linear1')
    for month,_,_ in folds():(root/'validation'/f'{month}_NSW1_lgb_current.json').write_text(json.dumps({'mae_mw':1,'fit_seconds':1,'converged':True}))
    with self.assertRaises(AssertionError):select(root)
    self.assertEqual(json.loads((root/'selection.json').read_text()),locked)

 def test_calendar_splits_and_holdout(self):
  f=folds();self.assertEqual(len(f),12)
  self.assertEqual(f[-1][2],boundary('2026-08'))
  for (_,a,b),(_,c,d) in zip(f,f[1:]):self.assertEqual(b,c);self.assertLess(a,c)
  self.assertEqual({x['parameters']['linear_tree'] for x in CANDIDATES.values() if x['family']=='lightgbm'},{True,False})

 def test_vector_features_match_production(self):
  idx=pd.date_range('2025-01-01',periods=1000,freq='15min',tz='UTC');v=np.random.default_rng(42).normal(size=1000)
  x,y=design(v,idx)
  for i in [672,800,999]:np.testing.assert_allclose(x[i-672],features(v[:i],idx[i]),atol=1e-12)
  changed=v.copy();changed[800:]+=10000;x2,_=design(changed,idx)
  np.testing.assert_allclose(x2[:129],x[:129])

 def test_recursive_forecasts_never_use_future_actuals(self):
  class Booster:
   def predict(self,x,num_threads):return .5*x[:,0]+.5*x[:,2]
  class Model:booster_=Booster()
  idx=pd.date_range('2025-01-01',periods=1100,freq='15min',tz='UTC');s=pd.Series(np.arange(1100,dtype=float),index=idx)
  origins=pd.DatetimeIndex([idx[700],idx[796]])
  p=recursive_month(Model(),np.zeros(15),np.ones(15),s,origins)
  s.loc[idx[700]:]+=10000;q=recursive_month(Model(),np.zeros(15),np.ones(15),s,origins)
  np.testing.assert_allclose(p[0],q[0]);self.assertFalse(np.allclose(p[1],q[1]))

 def test_sarima_dynamic_equals_prefix_only_forecast(self):
  rng=np.random.default_rng(7);y=20+np.sin(np.arange(600)*2*np.pi/24)+rng.normal(0,.2,600)
  fit=SARIMAX(y[:504],order=(1,0,0),seasonal_order=(1,1,0,24),trend='n').fit(disp=False,maxiter=60,cov_type='none',low_memory=False)
  future=y[504:];p=sarima_month(fit,future,np.zeros(4))
  for i in range(4):
   causal=fit if i==0 else fit.extend(future[:i*24])
   np.testing.assert_allclose(p[i],np.repeat(causal.forecast(24),4),atol=1e-7)
  changed=future.copy();changed[48:]+=1000;q=sarima_month(fit,changed,np.zeros(4))
  np.testing.assert_allclose(p[:3],q[:3],atol=1e-7)

if __name__=='__main__':unittest.main()
