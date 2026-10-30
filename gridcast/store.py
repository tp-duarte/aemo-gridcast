"""Append-only forecast ledger and revision-aware actuals in SQLite."""
import json
import sqlite3
from pathlib import Path
import pandas as pd
from .data import stamp, utc, STEP

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS raw_actuals(
 start TEXT, end TEXT, seen_at TEXT, value REAL NOT NULL, PRIMARY KEY(start, seen_at));
CREATE TABLE IF NOT EXISTS actuals(
 target TEXT, seen_at TEXT, value REAL NOT NULL, PRIMARY KEY(target, seen_at));
CREATE TABLE IF NOT EXISTS runs(
 id TEXT PRIMARY KEY, kind TEXT NOT NULL, issued_at TEXT NOT NULL,
 train_cutoff TEXT NOT NULL, created_at TEXT NOT NULL, metadata TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS forecasts(
 id INTEGER PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(id),
 model TEXT NOT NULL, target TEXT NOT NULL, horizon INTEGER NOT NULL,
 prediction REAL NOT NULL, lower REAL NOT NULL, upper REAL NOT NULL,
 UNIQUE(run_id, model, target));
CREATE TABLE IF NOT EXISTS scores(
 forecast_id INTEGER REFERENCES forecasts(id), actual_seen_at TEXT,
 scored_at TEXT NOT NULL, actual REAL NOT NULL, error REAL NOT NULL,
 PRIMARY KEY(forecast_id, actual_seen_at));
CREATE TABLE IF NOT EXISTS events(
 id INTEGER PRIMARY KEY, created_at TEXT NOT NULL, status TEXT NOT NULL, details TEXT NOT NULL);
CREATE VIEW IF NOT EXISTS latest_actuals AS
 SELECT a.* FROM actuals a WHERE a.seen_at=(SELECT MAX(b.seen_at) FROM actuals b WHERE b.target=a.target);
CREATE VIEW IF NOT EXISTS latest_scores AS
 SELECT s.* FROM scores s WHERE s.actual_seen_at=(
 SELECT MAX(t.actual_seen_at) FROM scores t WHERE t.forecast_id=s.forecast_id);
"""


class Store:
    def __init__(self, path, source=None):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        if source:
            old = self.db.execute("SELECT value FROM metadata WHERE key='source'").fetchone()
            if old and old[0] != source:
                self.db.close()
                raise ValueError("Use separate databases for different sources")
            with self.db:
                self.db.execute("INSERT OR IGNORE INTO metadata VALUES('source',?)", (source,))

    def close(self):
        self.db.close()

    def event(self, status, details):
        with self.db:
            self.db.execute("INSERT INTO events(created_at,status,details) VALUES(?,?,?)",
                            (stamp(utc()), status, json.dumps(details, allow_nan=False)))

    def raw(self, rows, seen_at):
        with self.db:
            for row in rows:
                t, end, value = stamp(row['startTime']), stamp(row['endTime']), float(row['value'])
                old = self.db.execute("SELECT value,end FROM raw_actuals WHERE start=? ORDER BY seen_at DESC LIMIT 1", (t,)).fetchone()
                if not old or old[0] != value or old[1] != end:
                    self.db.execute("INSERT OR IGNORE INTO raw_actuals VALUES(?,?,?,?)", (t, end, stamp(seen_at), value))

    def actuals(self, series, seen_at):
        with self.db:
            for target, value in series.items():
                old = self.db.execute("SELECT value FROM latest_actuals WHERE target=?", (stamp(target),)).fetchone()
                if old is None or old[0] != float(value):
                    self.db.execute("INSERT OR IGNORE INTO actuals VALUES(?,?,?)", (stamp(target), stamp(seen_at), float(value)))

    def series(self, as_of=None):
        if as_of is None:
            rows = self.db.execute("SELECT target,value FROM latest_actuals ORDER BY target").fetchall()
        else:
            rows = self.db.execute("""SELECT a.target,a.value FROM actuals a WHERE a.seen_at<=?
                AND a.seen_at=(SELECT MAX(b.seen_at) FROM actuals b WHERE b.target=a.target AND b.seen_at<=?)
                ORDER BY a.target""", (stamp(as_of), stamp(as_of))).fetchall()
        return pd.Series([r[1] for r in rows], index=pd.to_datetime([r[0] for r in rows], utc=True), dtype=float)

    def save_run(self, run_id, kind, issued, cutoff, metadata, predictions):
        with self.db:
            if self.db.execute("SELECT 1 FROM runs WHERE id=?", (run_id,)).fetchone():
                return False
            self.db.execute("INSERT INTO runs VALUES(?,?,?,?,?,?)", (run_id, kind, stamp(issued), stamp(cutoff), stamp(utc()), json.dumps(metadata, allow_nan=False)))
            for model, frame in predictions.items():
                for target, row in frame.iterrows():
                    self.db.execute("INSERT INTO forecasts(run_id,model,target,horizon,prediction,lower,upper) VALUES(?,?,?,?,?,?,?)",
                                    (run_id, model, stamp(target), int(row.horizon), float(row.prediction), float(row.lower), float(row.upper)))
        return True

    def score(self, now):
        """Score mature actuals; retain previous scores when actuals revise."""
        with self.db:
            before = self.db.total_changes
            self.db.execute("""INSERT OR IGNORE INTO scores
                SELECT f.id,a.seen_at,?,a.value,f.prediction-a.value
                FROM forecasts f JOIN latest_actuals a ON a.target=f.target
                WHERE a.target<=? AND a.seen_at<=?""",
                (stamp(now), stamp(utc(now) - STEP - pd.Timedelta(minutes=15)), stamp(now)))
            return self.db.total_changes - before

    def scored_frame(self, kind):
        return pd.read_sql_query("""SELECT f.*,r.kind,r.issued_at,s.actual,s.error,s.actual_seen_at
            FROM forecasts f JOIN runs r ON r.id=f.run_id
            JOIN latest_scores s ON s.forecast_id=f.id WHERE r.kind=?""", self.db, params=(kind,))
