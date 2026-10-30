import unittest
import numpy as np
import pandas as pd
from gridcast.direct import design,training,predict
from gridcast.models import seasonal_baseline

class DirectTests(unittest.TestCase):
 def test_baseline_equivalence_and_no_future_features(self):
  index=pd.date_range('2023-07-01',periods=410*96,freq='15min',tz='UTC')
  series=pd.Series(500+20*np.sin(np.arange(len(index))/96)+np.arange(len(index))*.002,index=index)
  for o in [400*96,400*96+12,400*96+48]:
   x,b=design(series,np.full(96,o),np.arange(96));targets=pd.date_range(index[o],periods=96,freq='15min')
   expected,_=seasonal_baseline(series.iloc[:o],targets);np.testing.assert_allclose(b,expected,rtol=1e-9)
   changed=series.copy();changed.iloc[o:]=1e10
   other,c=design(changed,np.full(96,o),np.arange(96));np.testing.assert_allclose(x,other);np.testing.assert_allclose(b,c)
  a,y=training(series,index[405*96]);changed=series.copy();changed.loc[index[405*96]:]=1e10
  z,yy=training(changed,index[405*96]);np.testing.assert_array_equal(a,z);np.testing.assert_array_equal(y,yy)
 def test_zero_correction_is_baseline(self):
  class Model:
   def predict(self,x):return np.zeros(len(x))
  index=pd.date_range('2023-07-01',periods=400*96,freq='15min',tz='UTC');series=pd.Series(500.,index=index)
  np.testing.assert_allclose(predict(Model(),series,index[-1]+pd.Timedelta(minutes=15)),500)
