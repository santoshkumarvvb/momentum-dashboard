import io
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

st.set_page_config(page_title="Global Stage 2 Momentum Dashboard", page_icon="📈", layout="wide")
st.title("📈 Global Stage 2 Momentum Dashboard")
st.caption("India (All NSE / Nifty 500) • USA • Europe/UK • Stage 2 + 52-week-high + multi-period relative-strength momentum • Yahoo Finance")

NIFTY500_URL="https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv"
EUROPE=[("AZN.L","AstraZeneca","UK","GBP","Health Care"),("SHEL.L","Shell","UK","GBP","Energy"),("HSBA.L","HSBC Holdings","UK","GBP","Financials"),("ULVR.L","Unilever","UK","GBP","Consumer Staples"),("REL.L","RELX","UK","GBP","Industrials"),("LSEG.L","London Stock Exchange Group","UK","GBP","Financials"),("RR.L","Rolls-Royce Holdings","UK","GBP","Industrials"),("BA.L","BAE Systems","UK","GBP","Industrials"),("NG.L","National Grid","UK","GBP","Utilities"),("GSK.L","GSK","UK","GBP","Health Care"),("SAP.DE","SAP","Germany","EUR","Technology"),("SIE.DE","Siemens","Germany","EUR","Industrials"),("ALV.DE","Allianz","Germany","EUR","Financials"),("DTE.DE","Deutsche Telekom","Germany","EUR","Communication Services"),("MUV2.DE","Munich Re","Germany","EUR","Financials"),("RHM.DE","Rheinmetall","Germany","EUR","Industrials"),("AIR.PA","Airbus","France","EUR","Industrials"),("MC.PA","LVMH","France","EUR","Consumer Discretionary"),("OR.PA","L'Oreal","France","EUR","Consumer Staples"),("SU.PA","Schneider Electric","France","EUR","Industrials"),("SAF.PA","Safran","France","EUR","Industrials"),("TTE.PA","TotalEnergies","France","EUR","Energy"),("ASML.AS","ASML Holding","Netherlands","EUR","Technology"),("INGA.AS","ING Groep","Netherlands","EUR","Financials"),("PHIA.AS","Philips","Netherlands","EUR","Health Care"),("AD.AS","Ahold Delhaize","Netherlands","EUR","Consumer Staples"),("NOVN.SW","Novartis","Switzerland","CHF","Health Care"),("ROG.SW","Roche","Switzerland","CHF","Health Care"),("NESN.SW","Nestle","Switzerland","CHF","Consumer Staples"),("UBSG.SW","UBS Group","Switzerland","CHF","Financials"),("ABBN.SW","ABB","Switzerland","CHF","Industrials")]

with st.sidebar:
    st.header("Settings")
    market=st.selectbox("Market",["India — All NSE stocks","India — Nifty 500","USA — S&P 500","USA — Nasdaq 100","USA — Combined","Europe / UK","Global"])
    st.caption("Saved results reopen without downloading. Use Start / resume to continue an interrupted scan, or Refresh for new prices.")
    refresh_requested=st.button("🔄 Refresh market data",use_container_width=True)
    strategy=st.selectbox("Strategy",["EMA200 recovery → all-time-high breakout","Stage 2 only","Momentum ranking"])
    recovery_mode=strategy.startswith("EMA200 recovery")
    recovery_only=st.checkbox("Show recovery BUY / HOLD / EXIT signals only",True) if recovery_mode else False
    if recovery_mode: st.caption("Below EMA200 within the previous 6 calendar months, then a fresh intraday all-time high. Model exit when daily low touches EMA200. RSI and volume are informational only. Current-session signals are provisional; HOLD/EXIT refer to model positions, not your actual holdings.")
    only_stage2=True if strategy=="Stage 2 only" else st.checkbox("Require Stage 2",False)
    st.caption("Stage 2 uses your SMA50/150/200 template, rising SMA200 over 20 OR 80 OR 100 sessions, weekly 52-week extremes, price >10 and price × average 50-day volume >1,000,000. Global markets use their quoted price units.")
    high_filter=st.selectbox("52-week-high filter",["All","Within 25%","Within 10%","Within 5%","Within 2%"])
    min_rsi=st.slider("Minimum RSI(14)",0,80,0,5)
    min_relvol=st.select_slider("Minimum relative volume",options=[0.0,.75,1.0,1.25,1.5,2.0],value=0.0)


def normalize_nse_list(raw, segment):
    raw=raw.rename(columns=lambda c: str(c).strip().upper().replace(" ", "_"))
    if "SYMBOL" not in raw or "NAME_OF_COMPANY" not in raw:
        raise ValueError("NSE security list is missing symbol/company columns")
    symbols=raw["SYMBOL"].fillna("").astype(str).str.strip()
    out=pd.DataFrame({"Ticker":symbols+".NS","Company":raw["NAME_OF_COMPANY"].fillna("").astype(str).str.strip(),
                      "Sector":"","Country":"India","Currency":"INR","Segment":segment})
    return out[symbols.ne("")].drop_duplicates("Ticker")

@st.cache_data(ttl=86400,show_spinner=False)
def load_all_nse():
    sources=[
        ("Mainboard","https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"),
        ("SME","https://nsearchives.nseindia.com/emerge/corporates/content/SME_EQUITY_L.csv"),
    ]
    frames=[]
    for segment,url in sources:
        r=requests.get(url,headers={"User-Agent":"Mozilla/5.0","Accept":"text/csv,*/*"},timeout=30)
        r.raise_for_status()
        frames.append(normalize_nse_list(pd.read_csv(io.BytesIO(r.content)),segment))
    out=pd.concat(frames,ignore_index=True).drop_duplicates("Ticker").reset_index(drop=True)
    if out.empty: raise ValueError("NSE security lists were empty")
    return out

@st.cache_data(ttl=86400,show_spinner=False)
def load_india():
    try:
        r=requests.get(NIFTY500_URL,headers={"User-Agent":"Mozilla/5.0","Accept":"text/csv,text/plain,*/*"},timeout=20);r.raise_for_status();raw=pd.read_csv(io.BytesIO(r.content));sc=next(c for c in raw.columns if str(c).strip().lower()=="symbol");cc=next((c for c in raw.columns if "company" in str(c).lower()),sc);sec=next((c for c in raw.columns if "industry" in str(c).lower() or "sector" in str(c).lower()),None)
        return pd.DataFrame({"Ticker":raw[sc].astype(str).str.strip()+".NS","Company":raw[cc].astype(str),"Sector":raw[sec].astype(str) if sec else "","Country":"India","Currency":"INR"}).drop_duplicates("Ticker")
    except Exception:
        raw=pd.read_csv("tickers.csv");out=raw[["Ticker","Company","Sector"]].copy();out["Country"]="India";out["Currency"]="INR";return out
@st.cache_data(ttl=86400,show_spinner=False)
def load_sp500():
    r=pd.read_csv("sp500.csv");o=pd.DataFrame({"Ticker":r["Symbol"].astype(str).str.replace(".","-",regex=False),"Company":r["Security"],"Sector":r["GICS Sector"]});o["Country"]="USA";o["Currency"]="USD";return o.drop_duplicates("Ticker")
@st.cache_data(ttl=86400,show_spinner=False)
def load_nasdaq():
    r=pd.read_csv("nasdaq100.csv");o=pd.DataFrame({"Ticker":r["Ticker"].astype(str),"Company":r["Company"],"Sector":r.get("GICS_Sector","")});o["Country"]="USA";o["Currency"]="USD";return o.drop_duplicates("Ticker")
@st.cache_data(ttl=86400,show_spinner=False)
def load_europe():return pd.DataFrame(EUROPE,columns=["Ticker","Company","Country","Currency","Sector"])
def get_universe(n):
    if n=="India — All NSE stocks":return load_all_nse()
    if n=="India — Nifty 500":return load_india()
    if n=="USA — S&P 500":return load_sp500()
    if n=="USA — Nasdaq 100":return load_nasdaq()
    if n=="USA — Combined":return pd.concat([load_sp500(),load_nasdaq()]).drop_duplicates("Ticker")
    if n=="Europe / UK":return load_europe()
    return pd.concat([load_all_nse(),load_sp500(),load_nasdaq(),load_europe()]).drop_duplicates("Ticker")

def benchmark_for(country):
    if country=="India":return "^NSEI"
    if country=="USA":return "^GSPC"
    if country=="UK":return "^FTSE"
    if country=="Germany":return "^GDAXI"
    if country=="France":return "^FCHI"
    if country=="Netherlands":return "^AEX"
    if country=="Switzerland":return "^SSMI"
    return "^STOXX"

def period_return(s,days):
    s=s.dropna()
    if len(s)<=days:return np.nan
    return (float(s.iloc[-1])/float(s.iloc[-days-1])-1)*100

@st.cache_data(ttl=900,show_spinner=False)
def benchmark_returns(ticker):
    try:
        h=yf.download(ticker,period="2y",interval="1d",auto_adjust=True,progress=False,threads=False)
        if isinstance(h.columns,pd.MultiIndex):h.columns=h.columns.get_level_values(0)
        c=h["Close"].dropna().astype(float)
        return {63:period_return(c,63),126:period_return(c,126),252:period_return(c,252)}
    except Exception:return {63:np.nan,126:np.nan,252:np.nan}


def chartink_stage2(h):
    """User's cash Stage-2 template; Friday weeks include the current partial week."""
    close = h["Close"].astype(float)
    sma = {n: close.rolling(n).mean() for n in (50, 150, 200)}
    weekly = h[["High", "Low"]].resample("W-FRI").agg({"High": "max", "Low": "min"}).dropna()
    weekly = weekly.tail(52)
    row = h.iloc[-1]
    p = float(row["Close"])
    s50, s150, s200 = (float(sma[n].iloc[-1]) for n in (50, 150, 200))
    rising = any(
        len(h) > offset and pd.notna(sma[200].iloc[-offset-1])
        and s200 > float(sma[200].iloc[-offset-1])
        for offset in (20, 80, 100)
    )
    hi = float(weekly["High"].max())
    lo = float(weekly["Low"].min())
    avgvol50 = float(h["Volume"].astype(float).rolling(50).mean().iloc[-1])
    ready = len(weekly) == 52 and all(pd.notna(v) for v in (s50, s150, s200, avgvol50))
    passed = bool(ready and p > s50 > s150 > s200 and rising
                  and float(row["Low"]) >= 1.5 * lo
                  and float(row["High"]) >= 0.75 * hi
                  and p > 10 and p * avgvol50 > 1000000
                  and float(row["Volume"]) > 0)
    return {"Stage 2": passed, "SMA50": s50, "SMA150": s150, "SMA200": s200,
            "SMA200 Rising": bool(rising), "Stage 2 Weekly High": hi,
            "Stage 2 Weekly Low": lo, "Avg Volume 50": avgvol50,
            "Stage 2 History Ready": bool(ready)}


def recovery_ath_signals(h):
    """Daily model on all available, split-adjusted Yahoo history; not an order executor."""
    c=h["Close"].astype(float)
    ema=c.ewm(span=200,adjust=False,min_periods=200).mean()
    prior_high=h["High"].astype(float).cummax().shift(1)
    dates=pd.Series(h.index,index=h.index)
    last_below=dates.where(c < ema).ffill().shift(1)
    recent_flags=(last_below >= (h.index-pd.DateOffset(months=6))).to_numpy()
    touches=(h["Low"].astype(float)<=ema).to_numpy()
    breakouts=((h["High"].astype(float)>prior_high)&(c>ema)).to_numpy()
    active=False
    entry_date=""
    signal="WAIT"
    for i in range(len(h)):
        signal="HOLD" if active else "WAIT"
        if active and touches[i]:
            active=False
            signal="EXIT"
        elif not active and recent_flags[i] and breakouts[i]:
            if touches[i]:
                signal="AMBIGUOUS"
            else:
                active=True
                entry_date=str(h.index[i].date())
                signal="BUY"
    recent=bool(recent_flags[-1])
    return {"Recovery Signal":signal, "Below EMA200 in 6M":bool(recent),
            "Prior All-Time High":float(prior_high.iloc[-1]),
            "Recovery EMA200":float(ema.iloc[-1]),
            "Distance to EMA200 %":(float(c.iloc[-1])/float(ema.iloc[-1])-1)*100,
            "Model Entry Date":entry_date if active or signal=="EXIT" else "",
            "History From":str(h.index[0].date()), "Data Date":str(h.index[-1].date())}

def analyze_history(ticker, h, recovery_mode=False):
    if h is None or h.empty:return None,None
    if isinstance(h.columns,pd.MultiIndex):h.columns=h.columns.get_level_values(0)
    h=h.dropna(subset=["Close"]).copy()
    if len(h)<220:return None,h
    close,high,low,vol=[h[c].astype(float) for c in ["Close","High","Low","Volume"]]
    for n in [20,50,100,150,200]:h[f"EMA{n}"]=close.ewm(span=n,adjust=False).mean()
    for n in [50,150,200]:h[f"SMA{n}"]=close.rolling(n).mean()
    d=close.diff();g=d.clip(lower=0);l=-d.clip(upper=0);ag=g.ewm(alpha=1/14,min_periods=14,adjust=False).mean();al=l.ewm(alpha=1/14,min_periods=14,adjust=False).mean();rs=ag/al.replace(0,np.nan);h["RSI14"]=100-(100/(1+rs));h["AvgVol20"]=vol.rolling(20).mean();h["RelVol"]=vol/h["AvgVol20"]
    row=h.iloc[-1];p=float(row["Close"]);e20,e50,e100,e150,e200=[float(row[f"EMA{n}"]) for n in [20,50,100,150,200]];old=float(h["EMA200"].iloc[-21]);hi=float(high.tail(252).max());lo=float(low.tail(252).min());dist=(p/hi-1)*100;rsi=float(row["RSI14"]) if pd.notna(row["RSI14"]) else np.nan;rv=float(row["RelVol"]) if pd.notna(row["RelVol"]) else np.nan
    r3,r6,r12=period_return(close,63),period_return(close,126),period_return(close,252)
    stage_details=chartink_stage2(h);stage=stage_details["Stage 2"];stack=e20>e50>e100>e200;w25,w10,w5,w2=dist>=-25,dist>=-10,dist>=-5,dist>=-2;fresh=p>=hi*.999;vb=bool(pd.notna(rv) and rv>=1.5)
    return {**(recovery_ath_signals(h) if recovery_mode else {}),**stage_details,"Ticker":ticker,"Price":p,"EMA20":e20,"EMA50":e50,"EMA100":e100,"EMA150":e150,"EMA200":e200,"EMA Stack":stack,"EMA200 Rising":e200>old,"52W High":hi,"% From 52W High":dist,"Within 25%":w25,"Within 10%":w10,"Within 5%":w5,"Within 2%":w2,"Fresh 52W High":fresh,"RSI14":rsi,"Rel Volume":rv,"Volume Breakout":vb,"Stage 2":stage,"Return 3M %":r3,"Return 6M %":r6,"Return 12M %":r12},h


def extract_ticker_history(batch, ticker, single=False):
    if batch is None or batch.empty:
        return pd.DataFrame()
    if isinstance(batch.columns,pd.MultiIndex):
        for level in range(batch.columns.nlevels):
            if ticker in batch.columns.get_level_values(level):
                return batch.xs(ticker,axis=1,level=level).copy()
        return pd.DataFrame()
    return batch.copy() if single else pd.DataFrame()


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

df=pd.DataFrame(records);all_df=df.copy()
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
if recovery_mode and recovery_only:df=df[df["Recovery Signal"].isin(["BUY","HOLD","EXIT","AMBIGUOUS"])]
if only_stage2:df=df[df["Stage 2"]]
fm={"Within 25%":"Within 25%","Within 10%":"Within 10%","Within 5%":"Within 5%","Within 2%":"Within 2%"}
if high_filter in fm:df=df[df[fm[high_filter]]]
if min_rsi:df=df[df["RSI14"]>=min_rsi]
if min_relvol:df=df[df["Rel Volume"]>=min_relvol]
df=df.sort_values(["Global Momentum Score","RS 6M %","% From 52W High"],ascending=[False,False,False]).reset_index(drop=True)

if recovery_mode:
    df=df.assign(_order=df["Recovery Signal"].map({"BUY":0,"HOLD":1,"EXIT":2,"AMBIGUOUS":3,"WAIT":4})).sort_values(["_order","Distance to EMA200 %"]).drop(columns="_order").reset_index(drop=True)
c1,c2,c3,c4,c5=st.columns(5);c1.metric("Stocks scanned",len(all_df));c2.metric("Stage 2",int(all_df["Stage 2"].sum()));c3.metric("Within 5%",int(all_df["Within 5%"].sum()));c4.metric("Within 2%",int(all_df["Within 2%"].sum()));c5.metric("New recovery entries" if recovery_mode else "Breakouts",int((all_df["Action"]==("BUY" if recovery_mode else "BREAKOUT")).sum()))

if strategy=="Momentum ranking" and (market.startswith("USA") or market=="Global"):
    st.subheader("🇺🇸 Top USA Momentum")
    usa=all_df[all_df["Country"]=="USA"].sort_values(["Global Momentum Score","RS 6M %"],ascending=False).head(20)
    usa_cols=["Ticker","Company","Sector","Price","Global Momentum Score","Action","Stage 2","% From 52W High","RSI14","Rel Volume","Return 3M %","Return 6M %","Return 12M %","RS 3M %","RS 6M %","RS 12M %"]
    usa_show=usa[usa_cols].copy()
    for c in usa_show.columns:
        if c not in ["Ticker","Company","Sector","Action","Stage 2"]:usa_show[c]=pd.to_numeric(usa_show[c],errors="coerce").round(2)
    st.dataframe(usa_show,use_container_width=True,hide_index=True)
    st.caption("RS columns = stock return minus S&P 500 return over the same period. Ranking is systematic, not a buy recommendation.")

st.subheader(f"🔥 {market} — {strategy}")
cols=["Country","Ticker","Company","Sector","Currency","Price","Global Momentum Score","Action","Stage 2","% From 52W High","RSI14","Rel Volume","Return 3M %","Return 6M %","Return 12M %","RS 3M %","RS 6M %","RS 12M %","SMA50","SMA150","SMA200","SMA200 Rising","EMA20","EMA50","EMA100","EMA150","EMA200"]
if recovery_mode:
    cols=["Country","Ticker","Company","Currency","Price","Recovery Signal","Below EMA200 in 6M","Prior All-Time High","Recovery EMA200","Distance to EMA200 %","Model Entry Date","Data Date","History From"]
    st.caption("ATH means highest intraday high in all available Yahoo history, excluding the current bar. The recovery test uses a daily close below EMA200 within 6 calendar months before entry. Prices are split-adjusted by Yahoo; missing early history can affect ATH. Model signals are not actual holdings or trade fills. The old momentum score does not control this strategy.")
show=df[cols].copy()
for c in show.columns:
    if c not in ["Country","Ticker","Company","Sector","Currency","Action","Stage 2","SMA200 Rising","Recovery Signal","Below EMA200 in 6M","Model Entry Date","Data Date","History From"]:show[c]=pd.to_numeric(show[c],errors="coerce").round(2)
st.dataframe(show,use_container_width=True,hide_index=True);st.download_button("Download current ranking",data=df.to_csv(index=False).encode("utf-8"),file_name="global_momentum_ranking.csv",mime="text/csv")

if not df.empty:
    st.subheader("📊 Stock Detail");selected=st.selectbox("Select stock",df["Ticker"].tolist());row=df[df["Ticker"]==selected].iloc[0];sym={"INR":"₹","USD":"$","GBP":"£","EUR":"€","CHF":"CHF "}.get(row["Currency"],"");a,b,c,d=st.columns(4);a.metric("Price",f"{sym}{row['Price']:.2f}");b.metric("Recovery signal",row["Recovery Signal"]) if recovery_mode else b.metric("Global Momentum",f"{int(row['Global Momentum Score'])}/100");c.metric("6M Return",f"{row['Return 6M %']:.1f}%");d.metric("6M Relative Strength",f"{row['RS 6M %']:+.1f}%");st.line_chart(histories[selected].tail(180)[["Close","EMA200"] if recovery_mode else ["Close","SMA50","SMA150","SMA200"]],use_container_width=True);st.write(f"**Signal:** {row['Action']} | **Stage 2:** {'Yes' if row['Stage 2'] else 'No'} | **52W distance:** {row['% From 52W High']:.2f}% | **RSI:** {row['RSI14']:.1f} | **Relative volume:** {row['Rel Volume']:.2f}× | **Benchmark:** {row['Benchmark']}")

st.caption("Systematic momentum screening only: Stage 2, 52-week-high proximity, volume, RSI, 3/6/12-month returns and benchmark-relative strength. Not investment advice.")

