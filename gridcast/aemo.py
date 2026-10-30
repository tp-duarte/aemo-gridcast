"""AEMO source adapters. Original files remain available with checksums."""
import csv
import hashlib
import io
import json
import re
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from typing import Protocol
import numpy as np
import pandas as pd
from .data import REGIONS, MARKET_TZ, STEP, utc

MONTHLY = "https://www.aemo.com.au/aemo/data/nem/priceanddemand/"
DISPATCH = "https://www.nemweb.com.au/Reports/Current/DispatchIS_Reports/"


class ObservationSource(Protocol):
    def fetch(self) -> pd.DataFrame:
        """Return region, end (UTC interval end), value (MW)."""


def get_bytes(url):
    for attempt in range(4):
        try:
            with urlopen(Request(url, headers={"User-Agent":"Mozilla/5.0"}), timeout=45) as response:
                return response.read()
        except HTTPError as exc:
            if exc.code not in (429,500,502,503,504):
                raise
        except (TimeoutError,URLError):
            pass
        time.sleep(2**attempt)
    raise RuntimeError("AEMO source unavailable after retries")


def download_months(directory, months, regions=REGIONS):
    directory = Path(directory)
    raw = directory/"raw"
    raw.mkdir(parents=True,exist_ok=True)
    manifest=[]
    for month in months:
        if not re.fullmatch(r"20\d{4}",month):
            raise ValueError("Month format is YYYYMM")
        for region in regions:
            if region not in REGIONS:
                raise ValueError("Invalid region")
            name=f"PRICE_AND_DEMAND_{month}_{region}.csv"
            url=MONTHLY+name
            body=get_bytes(url)
            pd.read_csv(io.BytesIO(body), nrows=2)  # reject non-CSV failures
            (raw/name).write_bytes(body)
            manifest.append({"file":name,"url":url,"sha256":hashlib.sha256(body).hexdigest(),"retrieved_at":utc().isoformat()})
            time.sleep(.3)
    (directory/"source-manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    return manifest


def read_monthly(paths):
    frames=[]
    for path in paths:
        f=pd.read_csv(path)
        required={"REGION","SETTLEMENTDATE","TOTALDEMAND","PERIODTYPE"}
        if not required.issubset(f.columns):
            raise ValueError(f"Unexpected monthly schema: {Path(path).name}")
        f=f[f.PERIODTYPE=="TRADE"].copy()
        ends=pd.to_datetime(f.SETTLEMENTDATE,format="%Y/%m/%d %H:%M:%S").dt.tz_localize(MARKET_TZ).dt.tz_convert("UTC")
        frames.append(pd.DataFrame({"region":f.REGION,"end":ends,"value":pd.to_numeric(f.TOTALDEMAND,errors="raise")}))
    if not frames:
        raise ValueError("No AEMO monthly CSVs found")
    return validate(pd.concat(frames,ignore_index=True))


def validate(frame):
    frame=frame.copy()
    if not set(frame.region).issubset(REGIONS):
        raise ValueError("Unknown AEMO region")
    if not np.isfinite(frame.value).all():
        raise ValueError("Nonfinite demand observation")
    frame["end"]=pd.to_datetime(frame.end,utc=True)
    if ((frame.end.dt.minute%5)!=0).any() or (frame.end.dt.second!=0).any():
        raise ValueError("Observations must lie on the five-minute grid")
    duplicates=frame[frame.duplicated(["region","end"],keep=False)]
    if len(duplicates) and (duplicates.groupby(["region","end"]).value.nunique()>1).any():
        raise ValueError("Conflicting duplicates within one source batch")
    return frame.drop_duplicates(["region","end"]).sort_values(["region","end"])


def aggregate(frame, now=None):
    """Three contiguous 5-minute source labels, mapped to start-labelled 15m MW mean.
    TOTALDEMAND is a dispatch-demand series, not settlement-grade metered consumption.
    Negative values are retained, never clipped.
    """
    frame=validate(frame)
    if now is not None:
        frame=frame[frame.end<=utc(now)]
    frame=frame.copy()
    frame["start"]=frame.end-pd.Timedelta(minutes=5)
    frame["bucket"]=frame.start.dt.floor("15min")
    result={};quality={}
    for region in REGIONS:
        rows=frame[frame.region==region]
        accepted={}
        for bucket,g in rows.groupby("bucket"):
            expected=pd.date_range(bucket+pd.Timedelta(minutes=5),periods=3,freq="5min")
            if len(g)==3 and list(g.end)==list(expected):
                accepted[bucket]=float(g.value.mean())
        series=pd.Series(accepted,dtype=float).sort_index()
        series.index=pd.DatetimeIndex(series.index,tz="UTC")
        result[region]=series
        expected_count=len(pd.date_range(rows.bucket.min(),rows.bucket.max(),freq=STEP)) if len(rows) else 0
        quality[region]={"source_rows":len(rows),"accepted_bins":len(series),
                         "missing_or_incomplete_bins":expected_count-len(series),
                         "negative_bins":int((series<0).sum())}
    aligned=pd.concat(result,axis=1).dropna()
    result["NEM"]=aligned.sum(axis=1)  # no partial-region totals
    quality["NEM"]={"accepted_bins":len(aligned),"complete_regions_required":5}
    return result,quality


def parse_dispatch(body):
    rows=[];headers={}
    with zipfile.ZipFile(io.BytesIO(body)) as z:
        for name in z.namelist():
            if not name.lower().endswith(".csv"):
                continue
            for row in csv.reader(io.StringIO(z.read(name).decode("utf-8-sig"))):
                if len(row)<5 or row[1:3]!=["DISPATCH","REGIONSUM"]:
                    continue
                if row[0]=="I":
                    headers[tuple(row[1:4])]=row[4:]
                elif row[0]=="D":
                    fields=headers.get(tuple(row[1:4]))
                    if fields is None:
                        raise ValueError("REGIONSUM data before header")
                    d=dict(zip(fields,row[4:]))
                    if float(d["INTERVENTION"])!=0:
                        continue
                    end=pd.Timestamp(d["SETTLEMENTDATE"]).tz_localize(MARKET_TZ).tz_convert("UTC")
                    rows.append({"region":d["REGIONID"],"end":end,"value":float(d["TOTALDEMAND"])})
    if not rows:
        raise ValueError("No non-intervention regional demand in dispatch report")
    return validate(pd.DataFrame(rows))


class DispatchSource:
    """Recent dispatch report connector. Cache files; re-read recent revisions.
    Current directory retains only a short rolling window; gaps require backfill.
    """
    def __init__(self, cache, max_reports=24):
        self.cache=Path(cache);self.cache.mkdir(parents=True,exist_ok=True)
        if not 1<=max_reports<=288:raise ValueError('Recent dispatch catch-up is limited to 288 reports')
        self.max_reports=max_reports

    def fetch(self):
        html=get_bytes(DISPATCH).decode()
        names=sorted(set(re.findall(r"PUBLIC_DISPATCHIS_\d+_\d+\.zip",html,re.I)))[-self.max_reports:]
        if not names:
            raise ValueError("No dispatch files in AEMO directory")
        def read_report(name):
            path=self.cache/name
            if not path.exists():
                body=get_bytes(DISPATCH+name)
                parsed=parse_dispatch(body)  # never cache an invalid or partial download
                temporary=path.with_suffix('.tmp')
                temporary.write_bytes(body)
                temporary.replace(path)
                time.sleep(.25)
                return parsed
            return parse_dispatch(path.read_bytes())
        # Bounded requests reduce catch-up latency; map retains source order so
        # later reports still win even when their downloads finish earlier.
        with ThreadPoolExecutor(max_workers=3) as pool:
            frames=list(pool.map(read_report,names))
        # A later dispatch run/file supersedes an earlier report for the same label.
        return validate(pd.concat(frames).drop_duplicates(["region","end"],keep="last"))
