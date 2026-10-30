"""Regional forecasting orchestration: sample, replay and prospective worker."""
import hashlib
import json
import time
from contextlib import ExitStack
from pathlib import Path
import numpy as np
import pandas as pd
from .aemo import read_monthly, aggregate, DispatchSource
from .data import REGIONS, STEP, utc, stamp
from .models import MODELS, VERSION, TRAINING_DAYS, clean_training, predict_all
from .metrics import drift, leaderboard
from .store import Store

SERIES=(*REGIONS,"NEM")


def open_stores(root, stack):
    root=Path(root)
    stores={}
    for region in SERIES:
        s=Store(root/f"{region}.db","aemo-totaldemand")
        stack.callback(s.close);stores[region]=s
        with s.db:
            s.db.execute("INSERT OR IGNORE INTO metadata VALUES('region',?)",(region,))
    return stores


def ingest(stores, frame, seen_at):
    bundle,quality=aggregate(frame,seen_at)
    for region in REGIONS:
        f=frame[frame.region==region]
        records=[{"startTime":stamp(r.end-pd.Timedelta(minutes=5)),"endTime":stamp(r.end),"value":r.value}
                 for r in f.itertuples()]
        stores[region].raw(records,seen_at)
        stores[region].actuals(bundle[region],seen_at)
    stores["NEM"].actuals(bundle["NEM"],seen_at)
    return quality


def predict_bundle(series, origin, horizon=96, training_days=TRAINING_DAYS, weather_cache=None):
    outputs={};artifacts={};train_info={}
    if weather_cache:
        from .weather import WeatherArchive
        archive=WeatherArchive(weather_cache)
    for region in REGIONS:
        train,missing=clean_training(series[region],origin,days=training_days)
        weather=archive.features(region,train.index,pd.date_range(origin,periods=horizon,freq=STEP),origin) if weather_cache else None
        targets,pred,artifact=predict_all(train,horizon=horizon,weather=weather)
        outputs[region]=pred;artifacts[region]=artifact
        train_info[region]={"start":stamp(train.index[0]),"rows":len(train),"missing_fraction":missing,
                            "sha256":hashlib.sha256(train.to_csv().encode()).hexdigest()}
    outputs["NEM"]={m:sum(outputs[r][m] for r in REGIONS) for m in MODELS}
    artifacts["NEM"]={"version":VERSION,"method":"sum of five regional model forecasts",
                      "regional_training_hashes":{r:train_info[r]["sha256"] for r in REGIONS}}
    train_info["NEM"]={"start":stamp(origin-pd.Timedelta(days=training_days)),"rows":training_days*96,
                        "missing_fraction":max(t["missing_fraction"] for t in train_info.values()),
                        "sha256":hashlib.sha256(json.dumps(artifacts["NEM"],sort_keys=True).encode()).hexdigest()}
    return targets,outputs,artifacts,train_info


def calibrate_bundle(series, origin, horizon=96, training_days=TRAINING_DAYS, weather_cache=None):
    residuals={r:{m:[[] for _ in range(int(np.ceil(horizon/24)))] for m in MODELS} for r in SERIES}
    for days in range(7+int(np.ceil(horizon/96))-1,int(np.ceil(horizon/96))-1,-1):
        cutoff=origin-pd.Timedelta(days=days)
        targets,outputs,_,_=predict_bundle(series,cutoff,horizon,training_days,weather_cache)
        for region in SERIES:
            actual=series[region].reindex(targets).to_numpy()
            for model,values in outputs[region].items():
                for h,error in enumerate(abs(values-actual)):
                    if np.isfinite(error):residuals[region][model][h//24].append(float(error))
    widths={}
    for region,models in residuals.items():
        widths[region]={}
        for model,groups in models.items():
            if any(len(g)<48 for g in groups):
                raise ValueError("Insufficient held-out calibration observations")
            widths[region][model]=[float(np.sort(g)[min(len(g)-1,int(np.ceil((len(g)+1)*.9))-1)]) for g in groups]
    return widths


def issue(stores, series, origin, kind, widths, issued_at=None, horizon=96, training_days=TRAINING_DAYS, weather_cache=None):
    run_id=f"{kind}:{stamp(origin)}:{VERSION}:d{training_days}:"+("gfs" if weather_cache else "demand")
    if all(s.db.execute("SELECT 1 FROM runs WHERE id=?",(run_id,)).fetchone() for s in stores.values()):
        return False
    if any(s.db.execute("SELECT 1 FROM runs WHERE id=?",(run_id,)).fetchone() for s in stores.values()):
        raise RuntimeError("Incomplete regional bundle: inspect run before retrying; never mix vintages")
    started=time.monotonic()
    targets,predictions,artifacts,train_info=predict_bundle(series,origin,horizon,training_days,weather_cache)
    issued_at=utc() if kind=="live" else (origin if issued_at is None else utc(issued_at))
    keep=targets>=issued_at
    if not keep.any():raise ValueError("No future targets available")
    for region,store in stores.items():
        metadata={"region":region,"source":"AEMO TOTALDEMAND","unit":"MW","frequency":"15min",
                  "model_artifact":artifacts[region],"train_start":train_info[region]["start"],
                  "train_rows":train_info[region]["rows"],"training_sha256":train_info[region]["sha256"],
                  "training_missing_fraction":train_info[region]["missing_fraction"],
                  "duration_seconds":time.monotonic()-started,"drift":drift(series[region],origin),
                  "interval_widths":widths[region],"nominal_horizon":horizon,
                  "interval_method":"90% past residual quantile by 6-hour lead bucket; NEM calibrated on aggregate errors",
                  "availability":"prospective ingestion snapshot" if kind=="live" else "retrospective final-vintage simulation",
                  "market_timezone":"fixed UTC+10; no daylight saving","negative_values":"preserved"}
        frames={}
        for model,values in predictions[region].items():
            width=np.repeat(widths[region][model],24)[:horizon]
            frames[model]=pd.DataFrame({"prediction":values,"lower":values-width,"upper":values+width,
                "horizon":np.arange(1,horizon+1)},index=targets).loc[keep]
        store.save_run(run_id,kind,issued_at,origin,metadata,frames)
    return True


def build_sample(data_dir):
    root=Path(data_dir)
    frame=read_monthly(sorted((root/"raw").glob("PRICE_AND_DEMAND_*.csv")))
    now=utc()
    with ExitStack() as stack:
        stores=open_stores(root/"ledger",stack)
        quality=ingest(stores,frame,now)
        series={r:s.series() for r,s in stores.items()}
        end=min(series[r].index[-1]+STEP for r in SERIES)
        first=end-pd.Timedelta(days=7)
        print("Calibrating four models across five regions + NEM...",flush=True)
        widths=calibrate_bundle(series,first)
        for i in range(7):
            issue(stores,series,first+pd.Timedelta(days=i),"backtest",widths)
            print(f"Backtest origin {i+1}/7 saved",flush=True)
        # Overlapping genuine historical simulations, distinct from daily backtest metrics.
        for origin in pd.date_range(end-pd.Timedelta(days=1),end,freq="3h"):
            issue(stores,series,origin,"evolution",widths)
        issue(stores,series,end,"sample",widths)
        outlook_widths=calibrate_bundle(series,end,horizon=192)
        issue(stores,series,end,"outlook",outlook_widths,horizon=192)
        results={}
        for region,s in stores.items():
            s.score(now)
            results[region]=leaderboard(s.scored_frame("backtest"))
            s.event("ok",{"mode":"historical_sample","quality":quality[region],"source_end":stamp(end)})
        report={"data_start":stamp(min(frame.end)-pd.Timedelta(minutes=5)),
                "data_end":stamp(max(frame.end)),"raw_rows":len(frame),"quality":quality,
                "backtest_origins":7,"models":list(MODELS),"results":results}
        (root/"results.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    return report


def replay(data_dir):
    """Real sample, simulated arrival times. Proves delayed labels without claiming live results."""
    root=Path(data_dir)
    frame=read_monthly(sorted((root/"raw").glob("PRICE_AND_DEMAND_*.csv")))
    cutoff=frame.end.max()-pd.Timedelta(days=1)
    with ExitStack() as stack:
        stores=open_stores(root/"replay",stack)
        ingest(stores,frame[frame.end<=cutoff],cutoff)
        series={r:s.series(as_of=cutoff) for r,s in stores.items()}
        widths=calibrate_bundle(series,cutoff)
        issue(stores,series,cutoff,"replay",widths)
        before=stores["NEM"].db.execute("SELECT COUNT(*) FROM latest_scores").fetchone()[0]
        scored_before=sum(s.score(cutoff) for s in stores.values())
        predictions_before=[tuple(r) for r in stores["NEM"].db.execute("SELECT id,prediction FROM forecasts ORDER BY id")]
        later=frame.end.max()+pd.Timedelta(minutes=15)
        ingest(stores,frame[frame.end>cutoff],later)
        added=sum(s.score(later) for s in stores.values())
        predictions_after=[tuple(r) for r in stores["NEM"].db.execute("SELECT id,prediction FROM forecasts ORDER BY id")]
        report={"mode":"historical arrival replay","issued_at":stamp(cutoff),"actuals_arrived_at":stamp(later),
                "new_scores_before_actual_arrival":scored_before,"new_scores_after_arrival":added,
                "nem_total_scored":stores["NEM"].db.execute("SELECT COUNT(*) FROM latest_scores").fetchone()[0],
                "forecasts_unchanged":predictions_before==predictions_after,"already_scored_before_rerun":before}
        (root/"replay-proof.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
        return report


def live_cycle(data_dir,source=None,forecast_every_hours=0,training_days=TRAINING_DAYS,weather_cache=None):
    root=Path(data_dir);source=source or DispatchSource(root/"dispatch-cache")
    frame=source.fetch();received=utc()
    with ExitStack() as stack:
        stores=open_stores(root/"ledger",stack)
        quality=ingest(stores,frame,received)
        series={r:s.series(as_of=received) for r,s in stores.items()}
        end=min(series[r].index[-1]+STEP for r in SERIES)
        scored=sum(s.score(received) for s in stores.values())
        fresh=(received-end).total_seconds()
        if fresh>1800:raise ValueError("AEMO observations stale; forecast withheld")
        last=stores["NEM"].db.execute("SELECT MAX(train_cutoff) FROM runs WHERE kind='live'").fetchone()[0]
        if forecast_every_hours and last and end-pd.Timestamp(last)<pd.Timedelta(hours=forecast_every_hours):
            details={"forecast_created":False,"new_scores":scored,"freshness_seconds":fresh,
                     "quality":quality,"reason":"Observations saved; next forecast batch is not due"}
            for s in stores.values():s.event("ok",details)
            return details
        # Gaps between sample history and current data are detected before training.
        for region in REGIONS:clean_training(series[region],end,days=training_days)
        key=f"live:{stamp(end)}:{VERSION}:d{training_days}:"+("gfs" if weather_cache else "demand")
        if all(s.db.execute("SELECT 1 FROM runs WHERE id=?",(key,)).fetchone() for s in stores.values()):
            created=False
        else:
            created=issue(stores,series,end,"live",calibrate_bundle(series,end,training_days=training_days,weather_cache=weather_cache),training_days=training_days,weather_cache=weather_cache)
        details={"forecast_created":created,"new_scores":scored,"freshness_seconds":fresh,"quality":quality}
        for s in stores.values():s.event("ok",details)
        return details


def worker(data_dir,once=False):
    while True:
        try: print(json.dumps(live_cycle(data_dir)),flush=True)
        except Exception as exc:
            with ExitStack() as stack:
                for s in open_stores(Path(data_dir)/"ledger",stack).values():
                    s.event("error",{"message":str(exc)})
            if once:raise
            print(json.dumps({"status":"error","message":str(exc)}),flush=True)
        if once:return
        time.sleep(300)
