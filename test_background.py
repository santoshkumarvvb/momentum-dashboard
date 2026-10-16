import ast
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

try:
    import yfinance
except ImportError:
    sys.modules["yfinance"] = types.ModuleType("yfinance")
try:
    import requests
except ImportError:
    sys.modules["requests"] = types.ModuleType("requests")

import background_update as worker


def history():
    index = pd.bdate_range("2023-01-01", periods=600)
    c = np.linspace(30, 100, len(index))
    return pd.DataFrame({"Open": c, "High": c+1, "Low": c-1, "Close": c,
                         "Volume": 100000}, index=index)


class BackgroundTests(unittest.TestCase):
    def test_incremental_merge_replaces_overlap_without_losing_ath(self):
        old = history()
        old.iloc[5, old.columns.get_loc("High")] = 999
        incoming = old.tail(8).copy()
        incoming.iloc[-1, incoming.columns.get_loc("Close")] = 101
        merged = worker.merge_history(old, incoming)
        self.assertEqual(len(merged), len(old))
        self.assertEqual(merged["High"].max(), 999)
        self.assertEqual(merged["Close"].iloc[-1], 101)

    def test_database_roundtrip_and_incremental_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = worker.open_database(Path(tmp)/"prices.sqlite")
            h = history()
            conn.execute("INSERT INTO history VALUES (?,?)", ("SBC.NS", h.to_json(orient="split", date_format="iso")))
            conn.commit()
            calls = []
            def download(symbols, **kwargs):
                calls.append(kwargs)
                return pd.concat({s: h.tail(10) for s in symbols}, axis=1)
            with patch.object(worker.yf, "download", download, create=True), patch.object(worker.time,"sleep"):
                refreshed = worker.download_updates(conn, ["SBC.NS"])
            self.assertEqual(refreshed, {"SBC.NS"})
            self.assertIn("start", calls[0])
            self.assertNotIn("period", calls[0])
            self.assertEqual(len(worker.read_history(conn,"SBC.NS")),600)
            conn.close()

    def test_partial_session_is_excluded(self):
        h = history()
        now = pd.Timestamp(h.index[-1]).tz_localize("Asia/Kolkata") + pd.Timedelta(hours=12)
        self.assertEqual(len(worker.completed_sessions(h,"India",now)),len(h)-1)

    def test_json_is_standard_and_preserves_bools(self):
        value = worker.clean_json({"a": np.bool_(True),"b":np.nan,"c":np.int64(3)})
        self.assertEqual(json.loads(json.dumps(value,allow_nan=False)),{"a":True,"b":None,"c":3})

    def test_prepared_market_routing_without_price_downloads(self):
        source = Path("app.py").read_text()
        tree = ast.parse(source)
        node = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="published_market")
        node.decorator_list = []
        chart = json.loads(history().tail(2)[["Close"]].to_json(orient="split",date_format="iso"))
        def snapshot(region):
            names={"india":[("SBC.NS",False),("BHEL.NS",True)],"usa":[("AAPL",True)],"europe":[("AZN.L",True)]}
            return {"generated_at":"2026-10-16T18:00:00Z","requested":len(names[region]),
                    "entries":[{"record":{"Ticker":t,"Nifty 500":member,"S&P 500":member,"Nasdaq 100":member},"chart":chart} for t,member in names[region]]}
        ns={"pd":pd,"published_region":snapshot}
        exec(compile(ast.Module(body=[node],type_ignores=[]),"app.py","exec"),ns)
        self.assertEqual(len(ns["published_market"]("India — All NSE stocks")[0]),2)
        self.assertEqual(ns["published_market"]("India — Nifty 500")[0][0]["Ticker"],"BHEL.NS")
        self.assertEqual(len(ns["published_market"]("Global")[0]),4)


if __name__ == "__main__":
    unittest.main()
