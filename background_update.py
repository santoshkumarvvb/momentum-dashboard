"""Incremental end-of-day database builder. No Streamlit session is required."""
import argparse
import gzip
import io
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import time

import numpy as np
import pandas as pd
import yfinance as yf

from market_engine import (load_all_nse, load_india, load_sp500, load_nasdaq,
                           load_europe, analyze_history, extract_ticker_history,
                           benchmark_for, period_return)

REPO = "santoshkumarvvb/momentum-dashboard"
TAG = "market-data"
CHART_COLUMNS = ["Close", "EMA200", "SMA50", "SMA150", "SMA200"]
PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def gh(*args, check=True):
    return subprocess.run(["gh", *args, "--repo", REPO], check=check,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def restore_asset(name, directory):
    release = gh("release", "view", TAG, "--json", "assets")
    assets = json.loads(release.stdout)["assets"]
    if name not in {a["name"] for a in assets}:
        print(f"No previous {name}; initializing this region.", flush=True)
    else:
        # Never replace an existing database after a transient restore failure.
        gh("release", "download", TAG, "--pattern", name,
           "--dir", str(directory), "--clobber")
    return directory / name


def open_database(path):
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE IF NOT EXISTS history (ticker TEXT PRIMARY KEY, data TEXT NOT NULL)")
    return conn


def read_history(conn, ticker):
    row = conn.execute("SELECT data FROM history WHERE ticker=?", (ticker,)).fetchone()
    if not row:
        return pd.DataFrame()
    h = pd.read_json(io.StringIO(row[0]), orient="split")
    h.index = pd.to_datetime(h.index)
    return h


def normalize_history(h):
    if h is None or h.empty:
        return pd.DataFrame()
    h = h.copy().dropna(subset=["Close"])
    index = pd.to_datetime(h.index)
    if index.tz is not None:
        index = index.tz_localize(None)
    h.index = index.normalize()
    return h.loc[~h.index.duplicated(keep="last")].sort_index()


def merge_history(old, incoming):
    incoming = normalize_history(incoming)
    if incoming.empty:
        return old
    return normalize_history(pd.concat([old, incoming]))[PRICE_COLUMNS]


def clean_json(value):
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def completed_sessions(h, country, now=None):
    clocks = {"India": ("Asia/Kolkata", "15:45"), "USA": ("America/New_York", "16:15"),
              "UK": ("Europe/London", "16:45"), "Germany": ("Europe/Berlin", "17:45"),
              "France": ("Europe/Paris", "17:45"), "Netherlands": ("Europe/Amsterdam", "17:45"),
              "Switzerland": ("Europe/Zurich", "17:45")}
    zone, cutoff = clocks[country]
    now = (now if now is not None else pd.Timestamp.now(tz="UTC")).tz_convert(zone)
    if now.strftime("%H:%M") < cutoff:
        return h[h.index.date < now.date()]
    return h


def region_universe(region):
    if region == "india":
        u = load_all_nse()
        nifty = set(load_india()["Ticker"])
        u["Nifty 500"] = u["Ticker"].isin(nifty)
    elif region == "usa":
        sp, nasdaq = load_sp500(), load_nasdaq()
        u = pd.concat([sp, nasdaq]).drop_duplicates("Ticker").reset_index(drop=True)
        u["S&P 500"] = u["Ticker"].isin(set(sp["Ticker"]))
        u["Nasdaq 100"] = u["Ticker"].isin(set(nasdaq["Ticker"]))
    else:
        u = load_europe()
    return u


def download_updates(conn, tickers, full=False):
    existing = {x[0] for x in conn.execute("SELECT ticker FROM history")}
    groups = [(False, [t for t in tickers if t in existing and not full]),
              (True, [t for t in tickers if t not in existing or full])]
    refreshed = set()
    empty_batches = 0
    for bootstrap, symbols in groups:
        for start in range(0, len(symbols), 30):
            batch_symbols = symbols[start:start + 30]
            kwargs = {"period": "max"} if bootstrap else {
                "start": min(read_history(conn, t).index[-1] for t in batch_symbols)
                         - pd.Timedelta(days=10)}
            try:
                batch = yf.download(batch_symbols, interval="1d", auto_adjust=False,
                                    actions=True, threads=4, group_by="ticker",
                                    progress=False, timeout=20, **kwargs)
            except Exception as exc:
                print(f"Batch failed: {type(exc).__name__}", flush=True)
                batch = None
            if batch is None or batch.empty:
                empty_batches += 1
                if empty_batches >= 3:
                    print("Provider returned three empty batches; stopping downloads and retaining saved history.", flush=True)
                    return refreshed
                time.sleep(20)
                continue
            empty_batches = 0
            for ticker in batch_symbols:
                h = extract_ticker_history(batch, ticker, len(batch_symbols) == 1)
                if h.empty:
                    continue
                # A split/bonus changes older OHLC; replace that ticker's history.
                split = "Stock Splits" in h and h["Stock Splits"].fillna(0).ne(0).any()
                if not bootstrap and split:
                    try:
                        h = yf.Ticker(ticker).history(period="max", auto_adjust=False)
                    except Exception:
                        continue
                    old = pd.DataFrame()
                else:
                    old = read_history(conn, ticker)
                merged = merge_history(old, h)
                if merged.empty:
                    continue
                conn.execute("INSERT OR REPLACE INTO history VALUES (?,?)",
                             (ticker, merged.to_json(orient="split", date_format="iso", double_precision=15)))
                refreshed.add(ticker)
            conn.commit()
            print(f"{'Bootstrap' if bootstrap else 'Incremental'}: {min(start+30,len(symbols))}/{len(symbols)}; updated {len(refreshed)}", flush=True)
            time.sleep(0.5)
    return refreshed


def build_snapshot(conn, universe, refreshed):
    benchmarks = {}
    for country in universe["Country"].unique():
        bt = benchmark_for(country)
        old = read_history(conn, bt)
        kwargs = {"period": "2y"} if old.empty else {"start": old.index[-1] - pd.Timedelta(days=10)}
        try:
            h = yf.Ticker(bt).history(auto_adjust=False, **kwargs)
            merged = merge_history(old, h)
            if not merged.empty:
                conn.execute("INSERT OR REPLACE INTO history VALUES (?,?)",
                             (bt, merged.to_json(orient="split", date_format="iso", double_precision=15)))
                old = merged
        except Exception:
            pass
        old = completed_sessions(old, country) if not old.empty else old
        benchmarks[bt] = {days: period_return(old["Close"], days) if not old.empty else np.nan
                          for days in (63, 126, 252)}
    conn.commit()
    entries = []
    for meta in universe.to_dict("records"):
        ticker = meta["Ticker"]
        h = read_history(conn, ticker)
        if not h.empty:
            h = completed_sessions(h, meta["Country"])
        rec, chart = analyze_history(ticker, h, recovery_mode=True)
        if rec is None:
            continue
        rec.update(meta)
        bt = benchmark_for(rec["Country"])
        rec["Benchmark"] = bt
        rec["Updated This Run"] = ticker in refreshed
        for days, label in [(63,"3M"), (126,"6M"), (252,"12M")]:
            rec[f"RS {label} %"] = rec[f"Return {label} %"] - benchmarks[bt][days]
        entries.append({"record": clean_json(rec),
                        "chart": json.loads(chart.tail(180)[CHART_COLUMNS].to_json(
                            orient="split", date_format="iso", double_precision=10))})
    return {"schema": 1, "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "requested": len(universe), "usable": len(entries),
            "refreshed": len(refreshed), "entries": entries}


def run_region(region, workdir, publish=True, full=False):
    workdir.mkdir(parents=True, exist_ok=True)
    archive = restore_asset(f"prices-{region}.sqlite.gz", workdir) if publish else workdir/f"prices-{region}.sqlite.gz"
    db_path = workdir / f"prices-{region}.sqlite"
    if archive.exists() and not db_path.exists():
        with gzip.open(archive, "rb") as src, db_path.open("wb") as dest:
            shutil.copyfileobj(src, dest)
    universe = region_universe(region)
    conn = open_database(db_path)
    try:
        refreshed = download_updates(conn, universe["Ticker"].tolist(), full=full)
        snapshot = build_snapshot(conn, universe, refreshed)
    finally:
        conn.close()
    # Save the database even when an outage prevents publishing a new snapshot.
    # A later run can finish bootstrap instead of downloading successful symbols again.
    with db_path.open("rb") as src, gzip.open(archive, "wb", compresslevel=5) as dest:
        shutil.copyfileobj(src, dest)
    if publish:
        gh("release", "upload", TAG, str(archive), "--clobber")
    # Retain prior published results instead of replacing them with a provider outage.
    healthy = (snapshot["usable"] >= max(1, int(snapshot["requested"]*.5))
               and snapshot["refreshed"] >= max(1, int(snapshot["requested"]*.5)))
    if not healthy:
        raise RuntimeError(f"{region}: provider coverage too low to publish: "
                           f"{snapshot['usable']} usable, {snapshot['refreshed']} updated / {snapshot['requested']}")
    output = workdir / f"{region}.json.gz"
    with gzip.open(output, "wt", encoding="utf-8") as f:
        json.dump(snapshot, f, allow_nan=False, separators=(",", ":"))
    if publish:
        gh("release", "upload", TAG, str(output), "--clobber")
    print(f"{region}: published {snapshot['usable']} stocks; {snapshot['refreshed']} updated", flush=True)
    return snapshot


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", choices=["india", "usa", "europe"], required=True)
    parser.add_argument("--workdir", type=Path, default=Path(".background-data"))
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    run_region(args.region, args.workdir, full=args.full)
