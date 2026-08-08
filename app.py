import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="Global Stage 2 Momentum", page_icon="📈", layout="wide")
st.title("📈 Global Stage 2 Momentum Dashboard")
st.caption("India + S&P 500 + Nasdaq-100 • Yahoo Finance • Stage 2")

with st.sidebar:
    market = st.selectbox("Universe", ["India Watchlist", "S&P 500", "Nasdaq-100", "USA Combined"])
    refresh_minutes = st.selectbox("Auto refresh", [5,10,15,30,60], index=2)
    st_autorefresh(interval=refresh_minutes*60*1000, key="refresh")
    only_stage2 = st.checkbox("Stage 2 only", False)
    only_near_high = st.checkbox("Within 5% of 52W high only", False)

@st.cache_data(ttl=86400)
def load_sp500():
    t = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
    df = pd.DataFrame({
        "Ticker": t["Symbol"].astype(str).str.replace(".", "-", regex=False),
        "Company": t["Security"].astype(str),
        "Sector": t["GICS Sector"].astype(str)
    })
    return df

@st.cache_data(ttl=86400)
def load_nasdaq100():
    tables = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100")
    for t in tables:
        cols = [str(c) for c in t.columns]
        if "Ticker" in cols and ("Company" in cols or "Company name" in cols):
            ticker_col = "Ticker"
            company_col = "Company" if "Company" in cols else "Company name"
            sector_col = next((c for c in t.columns if "Sector" in str(c)), None)
            return pd.DataFrame({
                "Ticker": t[ticker_col].astype(str).str.replace(".", "-", regex=False),
                "Company": t[company_col].astype(str),
                "Sector": t[sector_col].astype(str) if sector_col is not None else ""
            })
    raise RuntimeError("Nasdaq-100 table not found")

@st.cache_data(ttl=86400)
def load_india():
    return pd.read_csv("tickers.csv")[["Ticker","Company","Sector"]]

@st.cache_data(ttl=900, show_spinner=False)
def fetch_one(ticker):
    hist = yf.download(ticker, period="18mo", interval="1d", auto_adjust=False, progress=False, threads=False)
    if hist is None or hist.empty:
        return None, None
    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = hist.columns.get_level_values(0)
    hist = hist.dropna(subset=["Close"]).copy()
    if len(hist) < 220:
        return None, hist

    close = hist["Close"].astype(float)
    high = hist["High"].astype(float)
    low = hist["Low"].astype(float)
    vol = hist["Volume"].astype(float)

    for n in [20,50,150,200]:
        hist[f"EMA{n}"] = close.ewm(span=n, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    hist["RSI14"] = 100 - (100/(1+rs))
    hist["AvgVol20"] = vol.rolling(20).mean()
    hist["RelVol"] = vol / hist["AvgVol20"]

    r = hist.iloc[-1]
    c = float(r["Close"]); e20=float(r["EMA20"]); e50=float(r["EMA50"]); e150=float(r["EMA150"]); e200=float(r["EMA200"])
    e200_old = float(hist["EMA200"].iloc[-21])
    h52 = float(high.tail(252).max()); l52=float(low.tail(252).min())
    dist = (c/h52 - 1)*100
    rsi=float(r["RSI14"]); relvol=float(r["RelVol"])

    stage2 = c>e50 and e50>e150 and e150>e200 and e200>e200_old and c>=1.30*l52 and c>=0.75*h52
    ema_stack = e20>e50>e150>e200
    within2 = dist>=-2
    within5 = dist>=-5
    fresh_high = c>=h52*0.999
    volume_breakout = relvol>=1.5

    score = (30 if stage2 else 0)+(20 if ema_stack else 0)+(20 if within2 else (10 if within5 else 0))+(15 if volume_breakout else 0)+(10 if 60<=rsi<=75 else (5 if 55<=rsi<80 else 0))
    action = "BUY" if stage2 and ema_stack and within2 and volume_breakout else ("WATCH" if stage2 and within5 else "AVOID")

    return {
        "Ticker":ticker,"Price":c,"EMA20":e20,"EMA50":e50,"EMA150":e150,"EMA200":e200,
        "EMA Stack":ema_stack,"EMA200 Rising":e200>e200_old,"52W High":h52,"% From 52W High":dist,
        "RSI14":rsi,"Rel Volume":relvol,"Stage 2":stage2,"Within 2%":within2,"Within 5%":within5,
        "Fresh 52W High":fresh_high,"Volume Breakout":volume_breakout,"Momentum Score":score,"Action":action
    }, hist

if market == "India Watchlist":
    universe = load_india()
elif market == "S&P 500":
    universe = load_sp500()
elif market == "Nasdaq-100":
    universe = load_nasdaq100()
else:
    universe = pd.concat([load_sp500(), load_nasdaq100()], ignore_index=True).drop_duplicates("Ticker")

st.write(f"**Universe:** {market} — {len(universe)} symbols")

records=[]; histories={}
with st.spinner("Loading market data..."):
    progress=st.progress(0)
    total=max(len(universe),1)
    for i,(_,u) in enumerate(universe.iterrows(),start=1):
        ticker=str(u["Ticker"]).strip()
        try:
            rec,hist=fetch_one(ticker)
            if rec:
                rec["Company"]=u.get("Company",ticker)
                rec["Sector"]=u.get("Sector","")
                records.append(rec); histories[ticker]=hist
        except Exception:
            pass
        progress.progress(i/total)
    progress.empty()

if not records:
    st.error("No market data loaded.")
    st.stop()

df=pd.DataFrame(records)
if only_stage2: df=df[df["Stage 2"]]
if only_near_high: df=df[df["Within 5%"]]
df=df.sort_values(["Momentum Score","% From 52W High"],ascending=[False,False])

c1,c2,c3,c4=st.columns(4)
allr=pd.DataFrame(records)
c1.metric("Stocks scanned",len(records))
c2.metric("Stage 2",int(allr["Stage 2"].sum()))
c3.metric("Within 2%",int(allr["Within 2%"].sum()))
c4.metric("BUY signals",int((allr["Action"]=="BUY").sum()))

show=df[["Ticker","Company","Sector","Price","Momentum Score","Action","Stage 2","EMA Stack","% From 52W High","RSI14","Rel Volume","Fresh 52W High"]].copy()
for c in ["Price","% From 52W High","RSI14","Rel Volume"]:
    show[c]=pd.to_numeric(show[c],errors="coerce").round(2)

st.dataframe(show,use_container_width=True,hide_index=True)
st.download_button("Download ranking CSV",df.to_csv(index=False).encode("utf-8"),"momentum_ranking.csv","text/csv")

if len(df):
    selected=st.selectbox("Stock detail",df["Ticker"].tolist())
    row=df[df["Ticker"]==selected].iloc[0]
    a,b,c,d=st.columns(4)
    a.metric("Price",f"{row['Price']:.2f}")
    b.metric("Score",f"{int(row['Momentum Score'])}/100")
    c.metric("% from 52W high",f"{row['% From 52W High']:.2f}%")
    d.metric("RSI",f"{row['RSI14']:.1f}")
    st.line_chart(histories[selected].tail(180)[["Close","EMA20","EMA50","EMA200"]],use_container_width=True)

st.caption("Educational screening tool only; not investment advice.")
