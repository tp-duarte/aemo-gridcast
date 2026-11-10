"""Publish dated views of immutable forecasts, scored with available actuals."""
import json
from pathlib import Path

import pandas as pd

from .data import utc
from .hourly import (encode, evaluated, load_frame, load_models,
                     public_payloads, wide)
from .object_store import R2Objects


def publish_archive(objects, work):
    work = Path(work)
    now = utc()
    keys = objects.keys('forecasts/')
    if not keys:
        raise ValueError('No saved forecasts available')
    forecasts = pd.concat([load_frame(objects, key) for key in keys], ignore_index=True)
    forecasts = forecasts.drop_duplicates(['run_id', 'region', 'model', 'target'])
    actuals = load_frame(objects, 'state/actuals.parquet')
    scores = evaluated(forecasts, actuals)
    bundle, _ = load_models(objects, work/'models')
    reference = pd.read_parquet(work/'models/training_reference.parquet')
    # Historical drift needs its original training reference. Archive focuses on
    # forecast errors and retains each vintage's own model id and training cutoff.
    monitoring = {region: [] for region in [*wide(actuals).columns, 'NEM']}
    dates = sorted(forecasts.issued_at.dt.strftime('%Y-%m-%d').unique())
    revision = now.strftime('%Y%m%dT%H%M%S%f')
    for day in ['all', *dates]:
        selected = forecasts if day == 'all' else forecasts[forecasts.issued_at.dt.strftime('%Y-%m-%d') == day]
        matched = scores[scores.run_id.isin(selected.run_id)]
        payloads = public_payloads(actuals, selected, matched, bundle, now, reference, monitoring)
        for region, payload in payloads.items():
            regional = selected[selected.region == region]
            regional_scores = matched[matched.region == region]
            payload['kind'] = 'archive'
            payload['available_modes'] = ['live', 'archive']
            payload['archive_date'] = day
            payload['archive_evaluated_at'] = actuals.seen_at.max().isoformat()
            payload['latest_run']['metadata'] = {}
            payload['latest_run']['train_cutoff'] = None
            payload['evolution']['kind'] = 'archive'
            vintages = []
            for run_id, vintage in regional.sort_values('issued_at').groupby('run_id', sort=False):
                vintages.append({'id': run_id, 'issued_at': str(vintage.issued_at.iloc[0]),
                                 'model_id': str(vintage.model_id.iloc[0]),
                                 'forecasts': vintage[['model', 'target', 'prediction', 'lower', 'upper']].assign(
                                     target=lambda frame: frame.target.map(lambda t: t.isoformat())).to_dict('records')})
            payload['evolution']['vintages'] = vintages
            payload['evolution']['limit'] = len(vintages)
            series = wide(actuals)
            series['NEM'] = series.sum(axis=1, min_count=5)
            observed = series[region].loc[regional.target.min():regional.target.max()].dropna()
            payload['evolution']['actuals'] = [{'target': t.isoformat(), 'value': float(value)} for t, value in observed.items()]
            payload['data_range'] = {'start': str(regional.issued_at.min()), 'end': str(regional.target.max())}
            objects.put(f'public/archive/{revision}/{day}/{region}.json', encode(payload))
            export = regional.merge(regional_scores[['run_id', 'model', 'target', 'actual', 'error', 'actual_seen_at']],
                                    on=['run_id', 'model', 'target'], how='left', validate='one_to_one')
            objects.put(f'public/archive/{revision}/{day}/{region}.csv', export.to_csv(index=False).encode())
    # Publish index last so readers only see complete generations.
    manifest = {'revision': revision, 'dates': dates, 'updated_at': now.isoformat(),
                'forecast_issues': int(forecasts.run_id.nunique()), 'date_basis': 'UTC issue date'}
    objects.put('public/archive/index.json', encode(manifest))
    return manifest


if __name__ == '__main__':
    print(json.dumps(publish_archive(R2Objects(), '.archive-work'), indent=2))

