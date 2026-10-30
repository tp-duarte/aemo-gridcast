import unittest
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from gridcast.monitoring import compare, monitoring
from gridcast.store import Store

class MonitoringTests(unittest.TestCase):
    def test_identical_and_shifted_distributions(self):
        a=np.arange(100,dtype=float)
        same=compare(a,a)
        self.assertEqual(same['ks_distance'],0)
        self.assertEqual(same['psi'],0)
        shifted=compare(a,a+1000)
        self.assertEqual(shifted['ks_distance'],1)
        self.assertEqual(shifted['outside_training_range_pct'],100)
        self.assertAlmostEqual(sum(b['current_pct'] for b in shifted['bins']),100)
    def test_constant_empty_nonfinite(self):
        self.assertIsNone(compare([2,2],[3,3])['standardized_mean_shift'])
        self.assertEqual(compare([1,2],[float('nan')])['status'],'awaiting_observations')
    def test_windows_and_gaps(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            s=Store(Path(tmp)/'test.db')
            index=pd.date_range('2026-08-01',periods=44*96,freq='15min',tz='UTC')
            cutoff=index[42*96]
            values=pd.Series(np.arange(len(index),dtype=float),index=index).drop(index[42*96+2])
            s.actuals(values,index[-1]+pd.Timedelta(days=1))
            s.save_run('test','backtest',cutoff,cutoff,{}, {})
            r=monitoring(s,'backtest')[0]
            self.assertEqual(r['comparisons']['demand']['reference_n'],4032)
            self.assertEqual(r['comparisons']['demand']['current_n'],191)
            self.assertEqual(r['comparisons']['ramp']['current_n'],190)
            self.assertGreater(r['current_missing_pct'],0)
            # As-of snapshots must not use observations received later.
            s.save_run('live','live',cutoff,cutoff,{}, {})
            self.assertEqual(monitoring(s,'live')[0]['status'],'training_unavailable')
            s.close()
if __name__=='__main__': unittest.main()
