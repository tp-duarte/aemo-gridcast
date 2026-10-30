"""One scheduled forecast cycle, guarded against overlapping local workers.

Run with `python -m gridcast.batch --data data`. No scheduler is installed here.
Persistent SQLite volume is required. Cloud Run's temporary disk is NOT suitable.
"""
import argparse
import json
from contextlib import contextmanager
from pathlib import Path
from .pipeline import live_cycle
from .models import TRAINING_DAYS


@contextmanager
def exclusive_run(data_dir):
    root=Path(data_dir);root.mkdir(parents=True,exist_ok=True)
    lock=root/'batch.lock'
    try:
        descriptor=lock.open('x')
    except FileExistsError as exc:
        raise RuntimeError('Batch already running, or stale batch.lock requires operator review') from exc
    try:
        descriptor.write('Single writer; remove only after confirming no worker is active.\n');descriptor.close()
        yield
    finally:
        descriptor.close();lock.unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser(description='Run one AEMO ingest, forecast-save and evaluation batch')
    parser.add_argument('--data',default='data')
    parser.add_argument('--forecast-every-hours',type=float,default=6,
                        help='Collect/score every invocation; issue forecasts only after this many hours')
    parser.add_argument('--training-days',type=int,default=TRAINING_DAYS)
    parser.add_argument('--weather-cache',help='Prepared Open-Meteo single-run archive; missing data blocks issuance')
    args=parser.parse_args()
    if args.training_days<29:parser.error('--training-days must be at least 29')
    if args.forecast_every_hours<=0:parser.error('--forecast-every-hours must be positive')
    with exclusive_run(args.data):
        result=live_cycle(args.data,forecast_every_hours=args.forecast_every_hours,training_days=args.training_days,weather_cache=args.weather_cache)
        print(json.dumps(result,indent=2))

if __name__=='__main__':main()
