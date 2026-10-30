import argparse
import json
from pathlib import Path
from .aemo import download_months
from .pipeline import build_sample,replay,worker
from .server import serve
from .store import Store
from .data import REGIONS,utc


def main():
    p=argparse.ArgumentParser(description="AEMO Gridcast")
    p.add_argument("command",choices=["download","build","replay","worker","score","serve"])
    p.add_argument("--data",default="data")
    p.add_argument("--months",nargs="+",default=[f"2025{m:02}" for m in range(8,13)]+[f"2026{m:02}" for m in range(1,10)])
    p.add_argument("--once",action="store_true")
    p.add_argument("--host",default="127.0.0.1")
    p.add_argument("--port",type=int,default=8000)
    a=p.parse_args()
    if a.command=="download":print(json.dumps(download_months(a.data,a.months),indent=2))
    elif a.command=="build":print(json.dumps(build_sample(a.data),indent=2))
    elif a.command=="replay":print(json.dumps(replay(a.data),indent=2))
    elif a.command=="worker":worker(a.data,a.once)
    elif a.command=="serve":serve(a.data,a.host,a.port)
    else:
        result={}
        for region in (*REGIONS,"NEM"):
            s=Store(Path(a.data)/"ledger"/f"{region}.db")
            try:result[region]=s.score(utc())
            finally:s.close()
        print(json.dumps(result))

if __name__=="__main__":main()
