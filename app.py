import io
import requests
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="Global Stage 2 Momentum Dashboard", page_icon="📈", layout="wide")
st.title("📈 Global Stage 2 Momentum Dashboard")
st.caption("India • USA • Europe/UK • Stage 2 + 52-week-high momentum • Yahoo Finance")

NIFTY500_URL = "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv"

# Large/liquid European momentum universe. Yahoo Finance suffixes identify the local listing.
EUROPE = [
    ("AZN.L", "AstraZeneca", "UK", "GBP", "Health Care"),
    ("SHEL.L", "Shell", "UK", "GBP", "Energy"),
    ("HSBA.L", "HSBC Holdings", "UK", "GBP", "Financials"),
    ("ULVR.L", "Unilever", "UK", "GBP", "Consumer Staples"),
    ("REL.L", "RELX", "UK", "GBP", "Industrials"),
    ("LSEG.L", "London Stock Exchange Group", "UK", "GBP", "Financials"),
    ("RR.L", "Rolls-Royce Holdings", "UK", "GBP", "Industrials"),
    ("BA.L", "BAE Systems", "UK", "GBP", "Industrials"),
    ("NG.L", "National Grid", "UK", "GBP", "Utilities"),
    ("GSK.L", "GSK", "UK", "GBP", "Health Care"),
    ("SAP.DE", "SAP", "Germany", "EUR", "Technology"),
    ("SIE.DE", "Siemens", "Germany", "EUR", "Industrials"),
    ("ALV.DE", "Allianz", "Germany", "EUR", "Financials"),
    ("DTE.DE", "Deutsche Telekom", "Germany", "EUR", "Communication Services"),
    ("MUV2.DE", "Munich Re", "Germany", "EUR", "Financials"),
    ("RHM.DE", "Rheinmetall", "Germany", "EUR", "Industrials"),
    ("AIR.PA", "Airbus", "France", "EUR", "Industrials"),
    ("MC.PA", "LVMH", "France", "EUR", "Consumer Discretionary"),
    ("OR.PA", "L'Oreal", "France", "EUR", "Consumer Staples"),
    ("SU.PA", "Schneider Electric", "France", "EUR", "Industrials"),
    ("SAF.PA", "Safran", "France", "EUR", "Industrials"),
    ("TTE.PA", "TotalEnergies", "France", "EUR", "Energy"),
    ("ASML.AS", "ASML Holding", "Netherlands", "EUR", "Technology"),
    ("INGA.AS", "ING Groep", "Netherlands", "EUR", "Financials"),
    ("PHIA.AS", "Philips", "Netherlands", "EUR", "Health Care"),
    ("AD.AS", "Ahold Delhaize", "Netherlands", "EUR", "Consumer Staples"),
    ("NOVN.SW", "Novartis", "Switzerland", "CHF", "Health Care"),
    ("ROG.SW", "Roche", "Switzerland", "CHF", "Health Care"),
    ("NESN.SW", "Nestle", "Switzerland", "CHF", "Consumer Staples"),
    ("UBSG.SW", "UBS Group", "Switzerland", "CHF", "Financials"),
    ("ABBN.SW", "ABB", "Switzerland", "CHF", "Industrials"),
]

with st.sidebar:
    st.header("Settings")
    market = st.selectbox("Market", ["India — Nifty 500", "USA — S&P 500", "USA — Nasdaq 100", "USA — Combined", "Europe / UK", "Global"])
    refresh_minutes = st.selectbox("Auto refresh", [5, 10, 15, 30, 60], index=2)
    st_autorefresh(interval=refresh_minutes * 60 * 1000, key="market_refresh")
    only_stage2 = st.checkbox("Stage 2 only", value=False)
    high_filter = st.selectbox("52-week-high filter", ["All", "Within 25%", "Within 10%", "Within 5%", "Within 2%"])
    min_rsi = st.slider("Minimum RSI(14)", 0, 80, 0, 5)
    min_relvol = st.select_slider("Minimum relative volume", options=[0.0, 0.75, 1.0, 1.25, 1.5, 2.0], value=0.0)

@st.cache_data(ttl=86400, show_spinner=False)
def load_india():
    try:
        headers = {"User-Agent": "Mozilla/5.0", "Accept": "text/csv,text/plain,*/*"}
        r = requests.get(NIFTY500_URL, headers=headers, timeout=20)
        r.raise_for_status()
        raw = pd.read_csv(io.BytesIO(r.content))
        symbol_col = next(c for c in raw.columns if str(c).strip().lower() == "symbol")
        company_col = next((c for c in raw.columns if "company" in str(c).lower()), symbol_col)
        sector_col = next((c for c in raw.columns if "industry" in str(c).lower() or "sector" in str(c).lower()), None)
        return pd.DataFrame({"Ticker": raw[symbol_col].astype(str).str.strip() + ".NS", "Company": raw[company_col].astype(str), "Sector": raw[sector_col].astype(str) if sector_col else "", "Country": "India", "Currency": "INR"}).drop_duplicates("Ticker")
    except Exception:
        raw = pd.read_csv("tickers.csv")
        out = raw[["Ticker", "Company", "Sector"]].copy()
        out["Country"], out["Currency"] = "India", "INR"
        return out

@st.cache_data(ttl=86400, show_spinner=False)
def load_sp500():
    raw = pd.read_csv("sp500.csv")
    out = pd.DataFrame({"Ticker": raw["Symbol"].astype(str).str.replace(".", "-", regex=False), "Company": raw["Security"], "Sector": raw["GICS Sector"]})
    out["Country"], out["Currency"] = "USA", "USD"
    return out.drop_duplicates("Ticker")

@st.cache_data(ttl=86400, show_spinner=False)
def load_nasdaq():
    raw = pd.read_csv("nasdaq100.csv")
    out = pd.DataFrame({"Ticker": raw["Ticker"].astype(str), "Company": raw["Company"], "Sector": raw.get("GICS_Sector", "")})
    out["Country"], out["Currency"] = "USA", "USD"
    return out.drop_duplicates("Ticker")

@st.cache_data(ttl=86400, show_spinner=False)
def load_europe():
    return pd.DataFrame(EUROPE, columns=["Ticker", "Company", "Country", "Currency", "Sector"])

def get_universe(name):
    if name == "India — Nifty 500": return load_india()
    if name == "USA — S&P 500": return load_sp500()
    if name == "USA — Nasdaq 100": return load_nasdaq()
    if name == "USA — Combined": return pd.concat([load_sp500(), load_nasdaq()]).drop_duplicates("Ticker")
    if name == "Europe / UK": return load_europe()
    return pd.concat([load_india(), load_sp500(), load_nasdaq(), load_europe()]).drop_duplicates("Ticker")

@st.cache_data(ttl=900, show_spinner=False)
def fetch_one(ticker):
    hist = yf.download(ticker, period="18mo", interval="1d", auto_adjust=False, progress=False, threads=False)
    if hist is None or hist.empty: return None, None
    if isinstance(hist.columns, pd.MultiIndex): hist.columns = hist.columns.get_level_values(0)
    hist = hist.dropna(subset=["Close"]).copy()
    if len(hist) < 220: return None, hist
    close, high, low, volume = [hist[c].astype(float) for c in ["Close", "High", "Low", "Volume"]]
    for n in [20, 50, 100, 150, 200]: hist[f"EMA{n}"] = close.ewm(span=n, adjust=False).mean()
    delta = close.diff(); gain = delta.clip(lower=0); loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean(); avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan); hist["RSI14"] = 100 - (100 / (1 + rs))
    hist["AvgVol20"] = volume.rolling(20).mean(); hist["RelVol"] = volume / hist["AvgVol20"]
    row = hist.iloc[-1]; price = float(row["Close"])
    e20, e50, e100, e150, e200 = [float(row[f"EMA{n}"]) for n in [20, 50, 100, 150, 200]]
    e200_old = float(hist["EMA200"].iloc[-21]); high52 = float(high.tail(252).max()); low52 = float(low.tail(252).min())
    dist = (price / high52 - 1) * 100
    rsi = float(row["RSI14"]) if pd.notna(row["RSI14"]) else np.nan; rv = float(row["RelVol"]) if pd.notna(row["RelVol"]) else np.nan
    stage2 = price > e50 > e150 > e200 and e200 > e200_old and price >= 1.30 * low52 and price >= .75 * high52
    ema_stack = e20 > e50 > e100 > e200
    w25, w10, w5, w2 = dist >= -25, dist >= -10, dist >= -5, dist >= -2
    fresh = price >= high52 * .999; vol_break = bool(pd.notna(rv) and rv >= 1.5)
    score = (30 if stage2 else 0) + (20 if ema_stack else 0) + (20 if w2 else 10 if w5 else 7 if w10 else 5 if w25 else 0) + (15 if vol_break else 0) + (10 if pd.notna(rsi) and 60 <= rsi <= 75 else 5 if pd.notna(rsi) and 55 <= rsi < 80 else 0)
    action = "BUY" if stage2 and ema_stack and w2 and vol_break else "WATCH" if stage2 and w5 else "STAGE 2" if stage2 and w25 else "AVOID"
    return {"Ticker": ticker, "Price": price, "EMA20": e20, "EMA50": e50, "EMA100": e100, "EMA150": e150, "EMA200": e200, "EMA Stack": ema_stack, "EMA200 Rising": e200 > e200_old, "52W High": high52, "% From 52W High": dist, "Within 25%": w25, "Within 10%": w10, "Within 5%": w5, "Within 2%": w2, "Fresh 52W High": fresh, "RSI14": rsi, "Rel Volume": rv, "Volume Breakout": vol_break, "Stage 2": stage2, "Momentum Score": score, "Action": action}, hist

universe = get_universe(market).reset_index(drop=True)
st.info(f"Loaded {len(universe)} symbols for {market}. Market history is fetched from Yahoo Finance.")
records, histories = [], {}
with st.spinner("Downloading market data and calculating global momentum signals..."):
    progress = st.progress(0); total = max(len(universe), 1)
    for i, (_, stock) in enumerate(universe.iterrows(), 1):
        ticker = str(stock["Ticker"]).strip()
        try:
            rec, hist = fetch_one(ticker)
            if rec is not None:
                for c in ["Company", "Sector", "Country", "Currency"]: rec[c] = stock.get(c, "")
                records.append(rec); histories[ticker] = hist
        except Exception: pass
        progress.progress(i / total)
    progress.empty()
if not records: st.error("No usable Yahoo Finance data was returned. Please try again later."); st.stop()

df = pd.DataFrame(records); all_df = df.copy()
if only_stage2: df = df[df["Stage 2"]]
filter_map = {"Within 25%": "Within 25%", "Within 10%": "Within 10%", "Within 5%": "Within 5%", "Within 2%": "Within 2%"}
if high_filter in filter_map: df = df[df[filter_map[high_filter]]]
if min_rsi: df = df[df["RSI14"] >= min_rsi]
if min_relvol: df = df[df["Rel Volume"] >= min_relvol]
df = df.sort_values(["Momentum Score", "% From 52W High"], ascending=[False, False]).reset_index(drop=True)

c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Stocks scanned", len(all_df)); c2.metric("Stage 2", int(all_df["Stage 2"].sum())); c3.metric("Within 5%", int(all_df["Within 5%"].sum())); c4.metric("Within 2%", int(all_df["Within 2%"].sum())); c5.metric("BUY signals", int((all_df["Action"] == "BUY").sum()))
st.subheader(f"🔥 {market} Momentum Ranking")
cols = ["Country","Ticker","Company","Sector","Currency","Price","Momentum Score","Action","Stage 2","EMA20","EMA50","EMA100","EMA150","EMA200","% From 52W High","RSI14","Rel Volume","Fresh 52W High"]
show = df[cols].copy()
for c in ["Price","EMA20","EMA50","EMA100","EMA150","EMA200","% From 52W High","RSI14","Rel Volume"]: show[c] = pd.to_numeric(show[c], errors="coerce").round(2)
st.dataframe(show, use_container_width=True, hide_index=True)
st.download_button("Download current ranking", data=df.to_csv(index=False).encode("utf-8"), file_name="global_momentum_ranking.csv", mime="text/csv")

if not df.empty:
    st.subheader("📊 Stock Detail"); selected = st.selectbox("Select stock", df["Ticker"].tolist()); row = df[df["Ticker"] == selected].iloc[0]
    symbol = {"INR":"₹","USD":"$","GBP":"£","EUR":"€","CHF":"CHF "}.get(row["Currency"], "")
    a,b,c,d = st.columns(4); a.metric("Price", f"{symbol}{row['Price']:.2f}"); b.metric("Momentum Score", f"{int(row['Momentum Score'])}/100"); c.metric("% from 52W high", f"{row['% From 52W High']:.2f}%"); d.metric("RSI(14)", f"{row['RSI14']:.1f}")
    st.line_chart(histories[selected].tail(180)[["Close","EMA20","EMA50","EMA100","EMA200"]], use_container_width=True)
    st.write(f"**Signal:** {row['Action']} | **Stage 2:** {'Yes' if row['Stage 2'] else 'No'} | **Relative volume:** {row['Rel Volume']:.2f}× | **Country:** {row['Country']}")

st.caption("Stage 2 + 52-week-high momentum screening across India, USA and selected large/liquid European/UK shares. Yahoo Finance data. Educational screening tool only.")