"""Saved-vintage training versus non-overlapping post-cutoff observations."""
import json
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, cramervonmises_2samp, wasserstein_distance
from .models import clean_training
from .metrics import psi


def compare(reference, current):
    a, b = np.asarray(reference, dtype=float), np.asarray(current, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if not len(a) or not len(b):
        return {"status": "awaiting_observations", "reference_n": len(a), "current_n": len(b)}
    edges = np.linspace(min(a.min(), b.min()), max(a.max(), b.max()), 21)
    if edges[0] == edges[-1]:
        edges = np.linspace(a[0] - .5, a[0] + .5, 21)
    ah, bh = np.histogram(a, edges)[0], np.histogram(b, edges)[0]
    points = np.unique(np.r_[a, b])
    ks = np.max(np.abs(np.searchsorted(np.sort(a), points, side="right") / len(a)
                       - np.searchsorted(np.sort(b), points, side="right") / len(b)))
    sd = float(a.std())
    bandwidth=max(1e-6, sd*len(a)**(-.2), float(np.ptp(np.r_[a,b]))*.005)
    grid=np.linspace(min(a.min(),b.min())-3*bandwidth,max(a.max(),b.max())+3*bandwidth,160)
    def density(v):
        return np.exp(-.5*((grid[:,None]-v[None,:])/bandwidth)**2).mean(axis=1)/(bandwidth*np.sqrt(2*np.pi))
    da,db=density(a),density(b)
    ks_test=ks_2samp(a,b,method='asymp')
    cvm=cramervonmises_2samp(a,b,method='asymptotic') if min(len(a),len(b))>=2 else None
    tests=[{'name':'Kolmogorov–Smirnov','statistic':float(ks_test.statistic),'p_value':float(ks_test.pvalue)},
           {'name':'Cramér–von Mises','statistic':float(cvm.statistic) if cvm else None,'p_value':float(cvm.pvalue) if cvm else None}]

    return {"status": "ok", "reference_n": len(a), "current_n": len(b),
            "density": [{"x":float(x),"reference":float(y),"current":float(z)} for x,y,z in zip(grid,da,db)],
            "bandwidth_mw":bandwidth,"tests":tests,"wasserstein_mw":float(wasserstein_distance(a,b)),
            "psi": psi(a, b), "ks_distance": float(ks),
            "mean_shift_mw": float(b.mean() - a.mean()),
            "standardized_mean_shift": float((b.mean()-a.mean())/sd) if sd else None,
            "std_ratio": float(b.std()/sd) if sd else None,
            "outside_training_range_pct": float(100*np.mean((b<a.min()) | (b>a.max()))),
            "reference_mean": float(a.mean()), "current_mean": float(b.mean()),
            "reference_p95": float(np.quantile(a,.95)), "current_p95": float(np.quantile(b,.95)),
            "bins": [{"lower": float(edges[i]), "upper": float(edges[i+1]),
                      "reference_pct": float(100*ah[i]/len(a)), "current_pct": float(100*bh[i]/len(b))} for i in range(20)]}


def monitoring(store, kind):
    runs = store.db.execute("SELECT * FROM runs WHERE kind=? ORDER BY train_cutoff DESC LIMIT 16", (kind,)).fetchall()
    observed = store.series()
    result = []
    for run in reversed(runs):
        cutoff = pd.Timestamp(run["train_cutoff"])
        end = min(cutoff + pd.Timedelta(days=7), observed.index[-1]+pd.Timedelta(minutes=15)) if len(observed) else cutoff
        end = max(end, cutoff)
        current = observed.loc[(observed.index >= cutoff) & (observed.index < end)]
        source = store.series(as_of=run["issued_at"]) if kind == "live" else observed
        try:
            metadata=json.loads(run['metadata'])
            days=int(metadata.get('train_rows',4032)/96)
            train, missing = clean_training(source, cutoff, days=days)
            train.attrs = {}
        except ValueError:
            result.append({"run_id": run["id"], "status": "training_unavailable", "cutoff": cutoff.isoformat(), "comparisons": {}})
            continue
        def slots(index):
            local = index.tz_convert("Etc/GMT-10")
            return local.dayofweek*96 + local.hour*4 + local.minute//15
        typical = train.groupby(slots(train.index)).mean()
        ref_resid = train.to_numpy()-typical.reindex(slots(train.index)).to_numpy()
        cur_resid = current.to_numpy()-typical.reindex(slots(current.index)).to_numpy()
        # Reindex first: never interpret a gap spanning several bins as a 15-minute ramp.
        joined = pd.concat([train.tail(1), current]).reindex(pd.date_range(cutoff-pd.Timedelta(minutes=15), end-pd.Timedelta(minutes=15), freq="15min"))
        comparisons = {"demand": compare(train,current), "seasonal_residual": compare(ref_resid,cur_resid),
                       "ramp": compare(train.diff(),joined.diff().iloc[1:])}
        # Holm correction over the six tests for this vintage (3 features × 2 tests).
        family=[t for d in comparisons.values() for t in d.get('tests',[]) if t['p_value'] is not None]
        ordered=sorted(family,key=lambda t:t['p_value']);adjusted=0.
        for i,test in enumerate(ordered):
            adjusted=max(adjusted,min(1.,(len(ordered)-i)*test['p_value']))
            test['holm_p_value']=adjusted
        expected = int((end-cutoff)/pd.Timedelta(minutes=15))
        result.append({"run_id": run["id"], "cutoff": cutoff.isoformat(),
            "reference_start": train.index[0].isoformat(), "current_end": end.isoformat(),
            "status": "ready" if len(current)>=96 else "awaiting_observations" if not len(current) else "short_window",
            "training_missing_fraction": missing, "current_expected_n": expected,
            "current_missing_pct": 100*(1-len(current)/expected) if expected else None,
            "availability": "as-of ingestion training; latest observed revisions" if kind=="live" else "retrospective final-vintage data",
            "comparisons": comparisons})
    return result
