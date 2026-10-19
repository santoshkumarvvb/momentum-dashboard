import io
import gzip
import json
import sqlite3
import tempfile
import hashlib
import fcntl
from pathlib import Path
from contextlib import contextmanager
import requests
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from market_engine import get_universe, benchmark_for, period_return, analyze_history, extract_ticker_history

st.set_page_config(page_title="Global Stage 2 Momentum Dashboard", page_icon="📈", layout="wide")
st.title("📈 Global Stage 2 Momentum Dashboard")
st.caption("India (All NSE / Nifty 500) • USA • Europe/UK • Stage 2 + 52-week-high + multi-period relative-strength momentum • Yahoo Finance")


with st.sidebar:
    st.header("Settings")
    market=st.selectbox("Market",["India — All NSE stocks","India — Nifty 500","USA — S&P 500","USA — Nasdaq 100","USA — Combined","Europe / UK","Global"])
    use_background=st.checkbox("Use prepared daily results (fast)",True)
    st.caption("Prepared daily results update in the background. Manual scans remain available.")
    st.caption("Saved results reopen without downloading. Use Start / resume to continue an interrupted scan, or Refresh for new prices.")
    refresh_requested=st.button("🔄 Refresh market data",use_container_width=True)
    strategy=st.selectbox("Strategy",["EMA200 recovery → all-time-high breakout","Stage 2 only","Momentum ranking"])
    recovery_mode=strategy.startswith("EMA200 recovery")
    recovery_only=st.checkbox("Show recovery BUY / HOLD / EXIT signals only",True) if recovery_mode else False
    if recovery_mode: st.caption("Below EMA200 within the previous 6 calendar months, then a fresh intraday all-time high. Model exit when daily low touches EMA200. RSI and volume are informational only. Current-session signals are provisional; HOLD/EXIT refer to model positions, not your actual holdings.")
    only_stage2=True if strategy=="Stage 2 only" else st.checkbox("Require Stage 2 — must also pass (AND)",False,key="require_stage2")
    if recovery_mode and only_stage2:
        st.caption("Combined filter: show only recovery BUY/HOLD signals that also pass Stage 2. WAIT, EXIT and AMBIGUOUS rows are excluded.")
    st.caption("Stage 2 uses your SMA50/150/200 template, rising SMA200 over 20 OR 80 OR 100 sessions, weekly 52-week extremes, price >10 and price × average 50-day volume >1,000,000. Global markets use their quoted price units.")
    high_filter=st.selectbox("52-week-high filter",["All","Within 25%","Within 10%","Within 5%","Within 2%"])
    min_rsi=st.slider("Minimum RSI(14)",0,80,0,5)
    min_relvol=st.select_slider("Minimum relative volume",options=[0.0,.75,1.0,1.25,1.5,2.0],value=0.0)







@st.cache_data(ttl=900,show_spinner=False)
def benchmark_returns(ticker):
    try:
        h=yf.download(ticker,period="2y",interval="1d",auto_adjust=True,progress=False,threads=False)
        if isinstance(h.columns,pd.MultiIndex):h.columns=h.columns.get_level_values(0)
        c=h["Close"].dropna().astype(float)
        return {63:period_return(c,63),126:period_return(c,126),252:period_return(c,252)}
    except Exception:return {63:np.nan,126:np.nan,252:np.nan}









SCAN_DIR=Path(tempfile.gettempdir())/"momentum_dashboard_scans_v1"
SCAN_DIR.mkdir(parents=True,exist_ok=True)

def scan_key(tickers,recovery_mode):
    return hashlib.sha256(json.dumps([list(tickers),bool(recovery_mode),"v1"]).encode()).hexdigest()

@contextmanager
def scan_db():
    conn=sqlite3.connect(str(SCAN_DIR/"scans.sqlite3"),timeout=30)
    conn.execute("CREATE TABLE IF NOT EXISTS scans (key TEXT PRIMARY KEY, stamp TEXT, offset INTEGER, complete INTEGER)")
    conn.execute("CREATE TABLE IF NOT EXISTS results (key TEXT, ticker TEXT, record TEXT, history TEXT, PRIMARY KEY (key,ticker))")
    conn.execute("CREATE TABLE IF NOT EXISTS benchmarks (key TEXT, ticker TEXT, data TEXT, PRIMARY KEY (key,ticker))")
    try:
        with conn:
            yield conn
    finally:
        conn.close()

def load_saved_scan(key):
    with scan_db() as conn:
        state=conn.execute("SELECT stamp,offset,complete FROM scans WHERE key=?",(key,)).fetchone()
        if state is None:return None
        rows=conn.execute("SELECT ticker,record,history FROM results WHERE key=? ORDER BY rowid",(key,)).fetchall()
    records=[]
    histories={}
    for ticker,record,history in rows:
        records.append(json.loads(record))
        h=pd.read_json(io.StringIO(history),orient="split")
        h.index=pd.to_datetime(h.index)
        histories[ticker]=h
    return records,histories,state[0],int(state[1]),bool(state[2])

def save_scan_batch(key, records, histories, offset, complete):
    stamp=pd.Timestamp.now(tz="UTC").isoformat()
    def scalar(value):
        if isinstance(value,np.generic):return value.item()
        raise TypeError(f"Unsupported snapshot value: {type(value)}")
    rows=[(key,r["Ticker"],json.dumps(r,default=scalar),
           histories[r["Ticker"]].to_json(orient="split",date_format="iso",double_precision=15)) for r in records]
    with scan_db() as conn:
        conn.executemany("INSERT OR REPLACE INTO results VALUES (?,?,?,?)",rows)
        conn.execute("INSERT OR REPLACE INTO scans VALUES (?,?,?,?)",(key,stamp,offset,int(complete)))
    return stamp

def clear_saved_scan(key):
    with scan_db() as conn:
        conn.execute("DELETE FROM results WHERE key=?",(key,))
        conn.execute("DELETE FROM scans WHERE key=?",(key,))
        conn.execute("DELETE FROM benchmarks WHERE key=?",(key,))



def saved_benchmark(key,ticker):
    with scan_db() as conn:
        row=conn.execute("SELECT data FROM benchmarks WHERE key=? AND ticker=?",(key,ticker)).fetchone()
    if row:return {int(k):v for k,v in json.loads(row[0]).items()}
    values=benchmark_returns(ticker)
    with scan_db() as conn:
        conn.execute("INSERT OR REPLACE INTO benchmarks VALUES (?,?,?)",(key,ticker,json.dumps(values)))
    return values

def scan_prices(tickers, recovery_mode, _progress=None, reset=False):
    key=scan_key(tickers,recovery_mode)
    # Nonblocking file lock prevents a reconnected browser starting a duplicate job.
    with (SCAN_DIR/(key+".lock")).open("a") as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            st.info("This scan is already running. Reopen saved progress after the next batch; a duplicate download was not started.")
            st.stop()
        if reset:clear_saved_scan(key)
        saved=load_saved_scan(key)
        if saved:
            records,histories,stamp,offset,complete=saved
            if complete:return records,histories,stamp
        else:
            records,histories,offset=[],{},0
            stamp=save_scan_batch(key,[],{},0,False)
        batch_size=40
        for start in range(offset,len(tickers),batch_size):
            symbols=list(tickers[start:start+batch_size])
            previous_count=len(records)
            try:
                batch=yf.download(symbols,period="max" if recovery_mode else "2y",
                                  interval="1d",auto_adjust=False,progress=False,
                                  threads=4,group_by="ticker",timeout=12)
            except Exception:
                batch=None
            for ticker in symbols:
                try:
                    raw=extract_ticker_history(batch,ticker,len(symbols)==1)
                    rec,h=analyze_history(ticker,raw,recovery_mode)
                    if rec is not None:
                        records.append(rec)
                        histories[ticker]=h.tail(180).copy()
                except Exception:
                    continue
            done=min(start+batch_size,len(tickers))
            stamp=save_scan_batch(key,records[previous_count:],histories,done,done==len(tickers))
            if _progress is not None:
                _progress.progress(done/max(len(tickers),1),
                                   text=f"Saved {done:,} / {len(tickers):,} stocks; usable: {len(records):,}")
        return records,histories,stamp


@st.cache_data(ttl=300,show_spinner=False)
def published_region(region):
    url=f"https://github.com/santoshkumarvvb/momentum-dashboard/releases/download/market-data/{region}.json.gz"
    response=requests.get(url,timeout=45)
    response.raise_for_status()
    data=json.loads(gzip.decompress(response.content))
    if data.get("schema") != 1 or not data.get("entries"):
        raise ValueError("Prepared snapshot is empty or incompatible")
    return data

@st.cache_data(ttl=300,max_entries=8,show_spinner=False)
def published_market(market):
    regions=(["india","usa","europe"] if market=="Global" else
             ["india"] if market.startswith("India") else
             ["usa"] if market.startswith("USA") else ["europe"])
    records=[]
    histories={}
    stamps=[]
    requested=0
    for region in regions:
        data=published_region(region)
        stamps.append(data["generated_at"])
        requested+=data["requested"]
        for entry in data["entries"]:
            rec=entry["record"]
            if market=="India — Nifty 500" and not rec.get("Nifty 500"):continue
            if market=="USA — S&P 500" and not rec.get("S&P 500"):continue
            if market=="USA — Nasdaq 100" and not rec.get("Nasdaq 100"):continue
            records.append(rec)
            histories[rec["Ticker"]]=pd.DataFrame(entry["chart"]["data"],columns=entry["chart"]["columns"],
                                                 index=pd.to_datetime(entry["chart"]["index"]))
    if not records:raise ValueError("No prepared records for this market")
    return records,histories,stamps,requested

if use_background:
    if refresh_requested:
        published_region.clear()
        published_market.clear()
    try:
        with st.spinner("Loading prepared results..."):
            records,histories,stamps,requested=published_market(market)
    except Exception:
        st.info("Prepared results are not available yet. The first background build may still be running. You can use manual scanning while it completes.")
        st.link_button("View background update status","https://github.com/santoshkumarvvb/momentum-dashboard/actions/workflows/update-market-data.yml")
        st.stop()
    u=pd.DataFrame(records)
    oldest=min(pd.Timestamp(x) for x in stamps)
    st.caption(f"Prepared daily results: oldest regional update {oldest.tz_convert('Asia/Kolkata').strftime('%d %b %Y %H:%M IST')}. Showing {len(records):,} usable stocks. No stock-history downloads on this page.")
    if market in ["India — All NSE stocks","USA — Combined","Europe / UK","Global"]:
        st.caption(f"Universe coverage: {len(records):,} usable / {requested:,} listed symbols; insufficient or unavailable histories are excluded.")
    stale=sum(not bool(r.get("Updated This Run")) for r in records)
    if stale:st.warning(f"{stale:,} stocks used saved history because their latest download failed. Check Data Date before acting.")
    if pd.Timestamp.now(tz="UTC")-oldest > pd.Timedelta(hours=72):
        st.warning("Prepared results are more than 72 hours old. Check the background workflow status.")
else:
    try:
        u=get_universe(market).reset_index(drop=True)
    except Exception as exc:
        st.error(f"Could not load the selected stock universe: {exc}. Please refresh; All NSE never falls back to Nifty 500.")
        st.stop()
    st.info(f"Loaded {len(u)} symbols for {market}. Market history is fetched from Yahoo Finance.")
    tickers=tuple(u["Ticker"].astype(str).str.strip())
    saved=load_saved_scan(scan_key(tickers,recovery_mode))
    run_requested=st.button("▶ Start / resume scan",use_container_width=True)
    partial=False
    if refresh_requested or run_requested:
        if refresh_requested:benchmark_returns.clear()
        with st.spinner("Scanning and saving each batch. Interrupted scans can resume from the last saved batch."):
            prog=st.progress(0)
            records,histories,scan_time=scan_prices(tickers,recovery_mode,prog,reset=refresh_requested)
            prog.empty()
    elif saved:
        records,histories,scan_time,offset,complete=saved
        partial=not complete
        if partial:st.info(f"Restored partial scan: {offset:,} / {len(tickers):,} processed. Click Start / resume to continue.")
    else:
        st.info("Click Start / resume scan to download prices. Opening or reconnecting this page does not start a download.")
        st.stop()
    bench_cache={}
    metadata=u.set_index("Ticker")
    for rec in records:
        meta=metadata.loc[rec["Ticker"]]
        for field in ["Company","Sector","Country","Currency"]:
            rec[field]=meta.get(field,"")
        bt=benchmark_for(rec["Country"])
        if bt not in bench_cache:bench_cache[bt]=saved_benchmark(scan_key(tickers,recovery_mode),bt)
        br=bench_cache[bt]
        rec["Benchmark"]=bt
        for days,label in [(63,"3M"),(126,"6M"),(252,"12M")]:
            value=rec[f"Return {label} %"]
            rec[f"RS {label} %"]=value-br[days] if pd.notna(value) and pd.notna(br[days]) else np.nan
    st.caption(f"Last saved: {pd.Timestamp(scan_time).tz_convert('Asia/Kolkata').strftime('%d %b %Y %H:%M IST')}. Results stay unchanged until Refresh. Saved scans survive browser reconnects while this server retains its files.")
    st.caption(f"Usable price history: {len(records)} / {len(u)} symbols. " + ("Scan is incomplete; rankings are provisional." if partial else f"{len(u)-len(records)} skipped because history was missing, too short, or could not be downloaded."))
    if not records:
        st.warning("No usable records in this saved scan yet. Use Start / resume for a partial scan, or Refresh to retry completed downloads.")
        st.stop()


df=pd.DataFrame(records)
for column in ["Price","RSI14","Rel Volume","% From 52W High","Return 3M %","Return 6M %","Return 12M %",
               "RS 3M %","RS 6M %","RS 12M %","EMA20","EMA50","EMA100","EMA150","EMA200",
               "SMA50","SMA150","SMA200","Recovery EMA200","Prior All-Time High","Distance to EMA200 %"]:
    if column in df:df[column]=pd.to_numeric(df[column],errors="coerce")
if "Data Date" not in df:df["Data Date"]="Manual scan"
all_df=df.copy()
# Cross-sectional percentile ranks reward stocks outperforming peers without hard-coding return thresholds.
for col,out in [("Return 3M %","3M Rank"),("Return 6M %","6M Rank"),("Return 12M %","12M Rank")]:df[out]=df[col].rank(pct=True)*100
trend_points=np.where(df["Stage 2"],25,0)+np.where(df["EMA Stack"],10,0)
high_points=np.select([df["Within 2%"],df["Within 5%"],df["Within 10%"],df["Within 25%"]],[15,11,7,3],default=0)
volume_points=np.select([df["Rel Volume"]>=1.5,df["Rel Volume"]>=1.25,df["Rel Volume"]>=1.0],[10,7,4],default=0)
rsi_points=np.select([(df["RSI14"]>=60)&(df["RSI14"]<=75),(df["RSI14"]>=55)&(df["RSI14"]<80)],[10,5],default=0)
return_points=(df["3M Rank"].fillna(0)*.08+df["6M Rank"].fillna(0)*.10+df["12M Rank"].fillna(0)*.12)
rs_bonus=np.select([(df["RS 3M %"]>0)&(df["RS 6M %"]>0)&(df["RS 12M %"]>0),(df["RS 3M %"]>0)&(df["RS 6M %"]>0)],[10,5],default=0)
# Normalize to 100 because raw components can total 110.
df["Global Momentum Score"]=np.minimum(100,np.round((trend_points+high_points+volume_points+rsi_points+return_points+rs_bonus)*100/110)).astype(int)
df["Action"]=np.select([(df["Stage 2"])&(df["Within 2%"])&(df["Rel Volume"]>=1.5)&(df["RS 6M %"]>0),(df["Stage 2"])&(df["Within 5%"])&(df["RS 6M %"]>0),df["Stage 2"]],["BREAKOUT","WATCH","STAGE 2"],default="AVOID")
if recovery_mode:
    df["Stage 2 Action"]=df["Action"]
    df["Action"]=df["Recovery Signal"]
all_df=df.copy()
if recovery_mode and only_stage2:
    df=df[df["Stage 2"].eq(True) & df["Recovery Signal"].isin(["BUY","HOLD"])]
elif recovery_mode and recovery_only:
    df=df[df["Recovery Signal"].isin(["BUY","HOLD","EXIT","AMBIGUOUS"])]
elif only_stage2:
    df=df[df["Stage 2"].eq(True)]
fm={"Within 25%":"Within 25%","Within 10%":"Within 10%","Within 5%":"Within 5%","Within 2%":"Within 2%"}
if high_filter in fm:df=df[df[fm[high_filter]]]
if min_rsi:df=df[df["RSI14"]>=min_rsi]
if min_relvol:df=df[df["Rel Volume"]>=min_relvol]
df=df.sort_values(["Global Momentum Score","RS 6M %","% From 52W High"],ascending=[False,False,False]).reset_index(drop=True)

if recovery_mode:
    df=df.assign(_order=df["Recovery Signal"].map({"BUY":0,"HOLD":1,"EXIT":2,"AMBIGUOUS":3,"WAIT":4})).sort_values(["_order","Distance to EMA200 %"]).drop(columns="_order").reset_index(drop=True)
st.caption(f"Showing {len(df):,} matches from {len(all_df):,} stocks with usable data. Counters below follow all selected filters.")
c1,c2,c3,c4,c5=st.columns(5);c1.metric("Matching stocks",len(df));c2.metric("Stage 2",int(df["Stage 2"].sum()));c3.metric("Within 5%",int(df["Within 5%"].sum()));c4.metric("Within 2%",int(df["Within 2%"].sum()));c5.metric("New recovery entries" if recovery_mode else "Breakouts",int((df["Action"]==("BUY" if recovery_mode else "BREAKOUT")).sum()))

if strategy=="Momentum ranking" and (market.startswith("USA") or market=="Global"):
    st.subheader("🇺🇸 Top USA Momentum")
    usa=df[df["Country"]=="USA"].sort_values(["Global Momentum Score","RS 6M %"],ascending=False).head(20)
    usa_cols=["Ticker","Company","Sector","Price","Global Momentum Score","Action","Stage 2","% From 52W High","RSI14","Rel Volume","Return 3M %","Return 6M %","Return 12M %","RS 3M %","RS 6M %","RS 12M %"]
    usa_show=usa[usa_cols].copy()
    for c in usa_show.columns:
        if c not in ["Ticker","Company","Sector","Action","Stage 2"]:usa_show[c]=pd.to_numeric(usa_show[c],errors="coerce").round(2)
    st.dataframe(usa_show,use_container_width=True,hide_index=True)
    st.caption("RS columns = stock return minus S&P 500 return over the same period. Ranking is systematic, not a buy recommendation.")

st.subheader(f"🔥 {market} — {strategy}" + (" + Stage 2 (AND)" if recovery_mode and only_stage2 else ""))
cols=["Country","Ticker","Company","Sector","Currency","Price","Data Date","Global Momentum Score","Action","Stage 2","% From 52W High","RSI14","Rel Volume","Return 3M %","Return 6M %","Return 12M %","RS 3M %","RS 6M %","RS 12M %","SMA50","SMA150","SMA200","SMA200 Rising","EMA20","EMA50","EMA100","EMA150","EMA200"]
if recovery_mode:
    cols=["Country","Ticker","Company","Currency","Price","Recovery Signal","Stage 2","Below EMA200 in 6M","Prior All-Time High","Recovery EMA200","Distance to EMA200 %","Model Entry Date","Data Date","History From"]
    st.caption("ATH means highest intraday high in all available Yahoo history, excluding the current bar. The recovery test uses a daily close below EMA200 within 6 calendar months before entry. Prices are split-adjusted by Yahoo; missing early history can affect ATH. Model signals are not actual holdings or trade fills. The old momentum score does not control this strategy.")
show=df[cols].copy()
for c in show.columns:
    if c not in ["Country","Ticker","Company","Sector","Currency","Action","Stage 2","SMA200 Rising","Recovery Signal","Below EMA200 in 6M","Model Entry Date","Data Date","History From"]:show[c]=pd.to_numeric(show[c],errors="coerce").round(2)
st.dataframe(show,use_container_width=True,hide_index=True);st.download_button("Download current ranking",data=df.to_csv(index=False).encode("utf-8"),file_name="global_momentum_ranking.csv",mime="text/csv")

if not df.empty:
    st.subheader("📊 Stock Detail");selected=st.selectbox("Select stock",df["Ticker"].tolist());row=df[df["Ticker"]==selected].iloc[0];sym={"INR":"₹","USD":"$","GBP":"£","EUR":"€","CHF":"CHF "}.get(row["Currency"],"");a,b,c,d=st.columns(4);a.metric("Price",f"{sym}{row['Price']:.2f}");b.metric("Recovery signal",row["Recovery Signal"]) if recovery_mode else b.metric("Global Momentum",f"{int(row['Global Momentum Score'])}/100");c.metric("6M Return",f"{row['Return 6M %']:.1f}%");d.metric("6M Relative Strength",f"{row['RS 6M %']:+.1f}%");st.line_chart(histories[selected].tail(180)[["Close","EMA200"] if recovery_mode else ["Close","SMA50","SMA150","SMA200"]],use_container_width=True);st.write(f"**Signal:** {row['Action']} | **Stage 2:** {'Yes' if row['Stage 2'] else 'No'} | **52W distance:** {row['% From 52W High']:.2f}% | **RSI:** {row['RSI14']:.1f} | **Relative volume:** {row['Rel Volume']:.2f}× | **Benchmark:** {row['Benchmark']}")

st.caption("Systematic momentum screening only: Stage 2, 52-week-high proximity, volume, RSI, 3/6/12-month returns and benchmark-relative strength. Not investment advice.")

