import tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from gridcast.aemo import DispatchSource

class DispatchCacheTests(unittest.TestCase):
    def test_parallel_downloads_preserve_revision_order_and_reuse_cache(self):
        names=['PUBLIC_DISPATCHIS_202601010005_1.zip','PUBLIC_DISPATCHIS_202601010005_2.zip']
        def download(url):
            if url.endswith('/'):return (' '.join(names)).encode()
            if url.endswith(names[0]):time.sleep(.03);return b'old'
            return b'new'
        def parse(body):
            return pd.DataFrame([dict(region='NSW1',end=pd.Timestamp('2026-01-01T00:05Z'),value=1. if body==b'old' else 2.)])
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            with patch('gridcast.aemo.get_bytes',side_effect=download) as get,patch('gridcast.aemo.parse_dispatch',side_effect=parse):
                source=DispatchSource(directory)
                self.assertEqual(source.fetch().value.iloc[0],2.)
                self.assertEqual(get.call_count,3)
                self.assertEqual(source.fetch().value.iloc[0],2.)
                self.assertEqual(get.call_count,4)
                self.assertFalse(list(Path(directory).glob('*.tmp')))
    def test_invalid_download_is_not_cached(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            with patch('gridcast.aemo.get_bytes',side_effect=[b'PUBLIC_DISPATCHIS_202601010005_1.zip',b'bad']),patch('gridcast.aemo.parse_dispatch',side_effect=ValueError('Bad source')):
                with self.assertRaises(ValueError):DispatchSource(directory).fetch()
            self.assertEqual(list(Path(directory).iterdir()),[])
