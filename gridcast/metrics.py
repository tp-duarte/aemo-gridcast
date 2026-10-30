"""Forecast quality, interval calibration and descriptive drift diagnostics."""
import numpy as np
import pandas as pd


def metrics(frame):
    if frame.empty:
        return {}
    e = frame.prediction.to_numpy() - frame.actual.to_numpy()
    y = frame.actual.to_numpy()
    lo, hi = frame.lower.to_numpy(), frame.upper.to_numpy()
    denom = np.abs(y).sum()
    smape_denom = np.abs(y) + np.abs(frame.prediction.to_numpy())
    smape = np.divide(2 * np.abs(e), smape_denom, out=np.zeros_like(e), where=smape_denom != 0)
    interval_score = hi - lo + 20 * np.maximum(lo - y, 0) + 20 * np.maximum(y - hi, 0)
    return {"n": len(e), "mae_mw": float(np.abs(e).mean()),
            "rmse_mw": float(np.sqrt(np.mean(e ** 2))), "bias_mw": float(e.mean()),
            "wape_pct": float(100 * np.abs(e).sum() / denom) if denom else None,
            "smape_pct": float(100 * smape.mean()),
            "coverage_90_pct": float(100 * np.mean((y >= lo) & (y <= hi))),
            "mean_width_mw": float(np.mean(hi - lo)),
            "interval_score_90": float(interval_score.mean()),
            "p95_abs_error_mw": float(np.quantile(np.abs(e), .95))}


def leaderboard(frame):
    result = []
    for model, group in frame.groupby("model"):
        row = {"model": model, **metrics(group)}
        # Join identical issue/target pairs, never compare different support sets.
        base = frame[frame.model == "seasonal_naive"][["run_id", "target", "actual_seen_at", "error"]]
        paired = group.merge(base, on=["run_id", "target", "actual_seen_at"], suffixes=("", "_base"))
        denominator = paired.error_base.abs().mean()
        row["skill_vs_baseline_pct"] = (float(100 * (1 - paired.error.abs().mean() / denominator))
                                      if len(paired) and denominator > 0 else None)
        result.append(row)
    return sorted(result, key=lambda r: r["mae_mw"])


def psi(reference, current):
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, 11)))
    if len(edges) < 3:
        return None
    edges[0], edges[-1] = -np.inf, np.inf
    p = np.histogram(reference, edges)[0].astype(float) + .5
    q = np.histogram(current, edges)[0].astype(float) + .5
    p, q = p / p.sum(), q / q.sum()
    return float(np.sum((q - p) * np.log(q / p)))


def drift(series, origin):
    recent = series.loc[(series.index >= origin - pd.Timedelta(days=7)) & (series.index < origin)]
    reference = series.loc[(series.index >= origin - pd.Timedelta(days=35)) & (series.index < origin - pd.Timedelta(days=7))]
    if len(recent) < 500 or len(reference) < 2000:
        return {"status": "insufficient_data"}
    # Match hour-of-week seasonality before comparing distributions.
    def slots(index):
        local = index.tz_convert("Etc/GMT-10")
        return local.dayofweek * 96 + local.hour * 4 + local.minute // 15
    typical = reference.groupby(slots(reference.index)).mean()
    ref_resid = reference.to_numpy() - typical.reindex(slots(reference.index)).to_numpy()
    cur_resid = recent.to_numpy() - typical.reindex(slots(recent.index)).to_numpy()
    p = psi(ref_resid[np.isfinite(ref_resid)], cur_resid[np.isfinite(cur_resid)])
    return {"status": "investigate" if p is not None and p > .2 else "stable",
            "seasonal_residual_psi": p, "raw_load_psi": psi(reference, recent),
            "mean_shift_mw": float(recent.mean() - reference.mean()),
            "reference_n": len(reference), "recent_n": len(recent),
            "reference_days": 28, "recent_days": 7, "psi_threshold": .2}



def forecast_added_value(frame):
    """MAE savings on identical run/target/actual-vintage pairs; never currency."""
    result = []
    if frame.empty:
        return result
    keys = ["run_id", "target", "actual_seen_at"]
    for baseline in ("seasonal_naive",):
        base = frame[frame.model == baseline][keys + ["error"]]
        for model, group in frame.groupby("model"):
            paired = group.merge(base, on=keys, suffixes=("", "_base"), validate="one_to_one")
            if paired.empty:
                continue
            model_mae = float(paired.error.abs().mean())
            base_mae = float(paired.error_base.abs().mean())
            result.append({"model": model, "baseline": baseline, "n": len(paired),
                           "model_mae_mw": model_mae, "baseline_mae_mw": base_mae,
                           "mae_saved_mw": base_mae-model_mae,
                           "fav_pct": 100*(1-model_mae/base_mae) if base_mae > 0 else None})
    return result

def performance_history(frame):
    """Daily UTC target-date metrics, matched within each day; latest actual revisions."""
    if frame.empty:
        return []
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame.target, utc=True).dt.strftime("%Y-%m-%d")
    return [{"date": date, **row} for date, group in frame.groupby("date", sort=True)
            for row in forecast_added_value(group)]
