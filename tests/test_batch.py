import tempfile
import unittest
from pathlib import Path
from gridcast.batch import exclusive_run
from unittest.mock import MagicMock, patch
import pandas as pd
from gridcast.pipeline import live_cycle, SERIES


class BatchLockTests(unittest.TestCase):
    def setUp(self):
        self.temp_root=Path(__file__).resolve().parents[1]/".test-tmp"
        self.temp_root.mkdir(exist_ok=True)

    def test_second_writer_rejected_and_lock_released(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as directory:
            with exclusive_run(directory):
                with self.assertRaises(RuntimeError):
                    with exclusive_run(directory):
                        self.fail('Second writer entered')
            self.assertFalse((Path(directory)/'batch.lock').exists())

    def test_error_releases_current_lock(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as directory:
            with self.assertRaises(ValueError):
                with exclusive_run(directory):
                    raise ValueError('Source unavailable')
            with exclusive_run(directory):
                self.assertTrue((Path(directory)/'batch.lock').exists())

    def test_stale_lock_is_not_deleted(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as directory:
            lock=Path(directory)/'batch.lock';lock.write_text('existing')
            with self.assertRaises(RuntimeError):
                with exclusive_run(directory):
                    pass
            self.assertEqual(lock.read_text(),'existing')

    def test_collection_scores_without_refitting_before_next_batch(self):
        received=pd.Timestamp('2026-10-27T12:00:00Z')
        stores={region:MagicMock() for region in SERIES}
        for store in stores.values():
            store.series.return_value=pd.Series([100.0],index=[received-pd.Timedelta(minutes=15)])
            store.score.return_value=2
            store.db.execute.return_value.fetchone.return_value=('2026-10-27T10:00:00Z',)
        source=MagicMock()
        with patch('gridcast.pipeline.utc',return_value=received), patch('gridcast.pipeline.open_stores',return_value=stores), patch('gridcast.pipeline.ingest',return_value={}) as ingest, patch('gridcast.pipeline.calibrate_bundle') as calibrate, patch('gridcast.pipeline.issue') as issue:
            result=live_cycle(self.temp_root,source,forecast_every_hours=6)
        ingest.assert_called_once()
        self.assertFalse(result['forecast_created'])
        self.assertEqual(result['new_scores'],2*len(SERIES))
        calibrate.assert_not_called()
        issue.assert_not_called()
        for store in stores.values():
            store.score.assert_called_once_with(received)
            store.event.assert_called_once()
