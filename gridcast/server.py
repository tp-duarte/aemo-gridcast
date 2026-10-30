"""Versioned, read-only integration API and local dashboard."""
import json
import os
from contextlib import ExitStack
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse,parse_qs
import pandas as pd
from .data import NAMES,REGIONS,utc,STEP
from .metrics import leaderboard,metrics,forecast_added_value,performance_history
from .store import Store
from .monitoring import monitoring

MODES=("backtest","sample","live","outlook")
SERIES=(*REGIONS,"NEM")


def snapshot(data_dir,kind="backtest",region="NEM"):
    if region not in SERIES or kind not in MODES:raise ValueError("Invalid region or mode")
    root=Path(data_dir)
    store=Store(root/"ledger"/f"{region}.db")
    try:
        runs=store.db.execute("SELECT * FROM runs WHERE kind=? ORDER BY issued_at DESC LIMIT 1",(kind,)).fetchall()
        latest=dict(runs[0]) if runs else None
        forecasts=[]
        if latest:
            latest["metadata"]=json.loads(latest["metadata"])
            latest["metadata"].get("model_artifact",{}).get("lightgbm",{}).pop("model_text",None)
            forecasts=[dict(r) for r in store.db.execute("SELECT * FROM forecasts WHERE run_id=? ORDER BY target",(latest["id"],))]
        series=store.series()
        scored=store.scored_frame(kind)
        total=store.db.execute("SELECT COUNT(*) FROM forecasts f JOIN runs r ON r.id=f.run_id WHERE kind=?",(kind,)).fetchone()[0]
        events=[dict(r) for r in store.db.execute("SELECT * FROM events ORDER BY id DESC LIMIT 5")]
        for event in events:event["details"]=json.loads(event["details"])
        horizon=[]
        if len(scored):
            scored["bucket"]=((scored.horizon-1)//24+1)*6
            horizon=[{"model":m,"hours":int(h),**metrics(g)} for (m,h),g in scored.groupby(["model","bucket"])]
        evolution_kind="live" if kind=="live" else "evolution"
        vintages=[dict(r) for r in store.db.execute("SELECT id,issued_at,train_cutoff FROM runs WHERE kind=? ORDER BY issued_at DESC LIMIT 96",(evolution_kind,))][::-1]
        for v in vintages:
            v["forecasts"]=[dict(r) for r in store.db.execute("SELECT model,target,prediction,lower,upper FROM forecasts WHERE run_id=? ORDER BY target",(v["id"],))]
        targets=[f["target"] for v in vintages for f in v["forecasts"]]
        evolution_actuals=[]
        if targets:
            evolution_actuals=[dict(r) for r in store.db.execute("SELECT target,value FROM latest_actuals WHERE target>=? AND target<=? ORDER BY target",(min(targets),max(targets)))]
        # Include actuals aligned to the displayed vintage, with five days of actual context.
        if forecasts:
            left=pd.Timestamp(min(f["target"] for f in forecasts))-pd.Timedelta(days=5)
            right=pd.Timestamp(max(f["target"] for f in forecasts))
            displayed=series.loc[(series.index>=left)&(series.index<=right)].tail(480)
        else:displayed=series.tail(480)
        result={"schema_version":"1.0","source":"aemo","region":region,"region_name":NAMES[region],"kind":kind,
            "generated_at":utc().isoformat(),"data_updated_at":store.db.execute("SELECT MAX(seen_at) FROM actuals").fetchone()[0],
            "backtest_fav":forecast_added_value(store.scored_frame("backtest")),"latest_run":latest,"forecasts":forecasts,
            "leaderboard":leaderboard(scored),"by_horizon":horizon,"events":events,
            "forecast_added_value":forecast_added_value(scored),"performance_history":performance_history(scored),
            "monitoring":monitoring(store,kind),
            "forecast_count":total,"scored_count":len(scored),"pending_count":total-len(scored),
            "actuals":[{"target":t.isoformat(),"value":float(v)} for t,v in displayed.items()],
            "evolution":{"vintages":vintages,"actuals":evolution_actuals,"limit":96,"kind":evolution_kind},
            "data_range":{"start":series.index[0].isoformat() if len(series) else None,"end":(series.index[-1]+STEP).isoformat() if len(series) else None},
            "reconciliation":"NEM point forecasts equal the sum of five regional forecasts; intervals calibrated on NEM errors"}
        return result
    finally:store.close()


def full_bundle(data_dir,kind):
    result={r:snapshot(data_dir,kind,r) for r in SERIES}
    previews=[{"region":r,"name":NAMES[r],"forecasts":result[r]["forecasts"],"actuals":result[r]["actuals"],"backtest_fav":result[r]["backtest_fav"],"data_updated_at":result[r]["data_updated_at"],"latest_run":result[r]["latest_run"]} for r in REGIONS]
    for entry in result.values():entry["regional_previews"]=previews
    return result


def serve(data_dir,host="127.0.0.1",port=8000):
    allowed=os.environ.get("GRIDCAST_ALLOWED_ORIGIN","")
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url=urlparse(self.path);params=parse_qs(url.query)
            kind=params.get("mode",["backtest"])[0];region=params.get("region",["NEM"])[0]
            if kind not in MODES or region not in SERIES:
                self.send_error(400,"Invalid mode or region");return
            status=200
            try:
                if url.path in ("/", "/index.html"):
                    body=Path(__file__).with_name("dashboard.html").read_bytes();mime="text/html; charset=utf-8"
                elif url.path=="/methodology.html":
                    body=(Path(__file__).resolve().parent.parent/"methodology.html").read_bytes();mime="text/html; charset=utf-8"
                elif url.path=="/v1/regions":
                    body=json.dumps([{"id":r,"name":NAMES[r]} for r in SERIES]).encode();mime="application/json"
                elif url.path=="/v1/dashboard":
                    data=snapshot(data_dir,kind,region)
                    data["regional_previews"]=[{"region":r,"name":NAMES[r],
                        "forecasts":(d:=snapshot(data_dir,kind,r))["forecasts"],"actuals":d["actuals"],"backtest_fav":d["backtest_fav"],"data_updated_at":d["data_updated_at"],"latest_run":d["latest_run"]} for r in REGIONS]
                    body=json.dumps(data,allow_nan=False).encode();mime="application/json"
                elif url.path=="/v1/forecasts.csv":
                    s=Store(Path(data_dir)/"ledger"/f"{region}.db")
                    try:
                        frame=pd.read_sql_query("""SELECT r.kind,r.issued_at,r.train_cutoff,f.*,s.actual,s.error,s.actual_seen_at
                          FROM forecasts f JOIN runs r ON r.id=f.run_id LEFT JOIN latest_scores s ON s.forecast_id=f.id WHERE r.kind=?""",s.db,params=(kind,))
                        frame.insert(0,"region",region);body=frame.to_csv(index=False).encode()
                    finally:s.close()
                    mime="text/csv"
                elif url.path=="/health":
                    s=Store(Path(data_dir)/"ledger"/"NEM.db")
                    try:event=s.db.execute("SELECT * FROM events ORDER BY id DESC LIMIT 1").fetchone()
                    finally:s.close()
                    age=(utc()-pd.Timestamp(event["created_at"])).total_seconds() if event else None
                    ready=bool(event and event["status"]=="ok" and age<float(os.environ.get("GRIDCAST_HEALTH_MAX_AGE_SECONDS","900")))
                    body=json.dumps({"status":"ready" if ready else "idle_or_unready","worker_event_age_seconds":age}).encode()
                    status=200 if ready else 503;mime="application/json"
                else:self.send_error(404);return
                self.send_response(status)
                self.send_header("Content-Type",mime);self.send_header("Content-Length",str(len(body)))
                self.send_header("Cache-Control","no-store")
                if allowed and self.headers.get("Origin")==allowed:
                    self.send_header("Access-Control-Allow-Origin",allowed);self.send_header("Vary","Origin")
                self.end_headers();self.wfile.write(body)
            except Exception as exc:
                print(f"API error: {type(exc).__name__}: {exc}",flush=True)
                self.send_error(500,"Unable to load forecast data")
    print(f"Gridcast: http://{host}:{port}",flush=True)
    ThreadingHTTPServer((host,port),Handler).serve_forever()
