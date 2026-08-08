
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="Momentum Stage 2 Dashboard", page_icon="📈", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.1rem; padding-bottom: 2rem;}
[data-testid="stMetricValue"] {font-size: 1.8rem;}
.badge-buy {background:#d8f3dc; color:#14532d; padding:4px 10px; border-radius:999px; font-weight:700;}
.badge-watch {background:#fff3bf; color:#7c4a03; padding:4px 10px; border-radius:999px; font-weight:700;}
.badge-avoid {background:#ffe3e3; color:#7f1d1d; padding:4px 10px; border-radius:999px; font-weight:700;}
</style>
""", unsafe_allow_html=True)

st.title("📈 Stage 2 Momentum Dashboard")
st.caption("Yahoo Finance • EMA trend template • 52-week high proximity • RSI • Relative Volume")

with st.sidebar:
    st.header("Settings")
    refresh_minutes = st.selectbox("Auto refresh", [5, 10, 15, 30, 60], index=2)
    st_autorefresh(interval=refresh_minutes*60*1000, key="market_refresh")

    min_price = st.number_input("Minimum price (₹)", value=50.0, step=10.0)
    min_avg_value_cr = st.number_input("Minimum avg traded value (₹ Cr)", value=10.0, step=5.0)
    only_stage2 = st.checkbox("Show Stage 2 only", value=False)
    only_near_high = st.checkbox("Within 5% of 52W high only", value=False)
    st.divider()
    st.caption("Use NSE tickers with .NS suffix. Edit tickers.csv to change the universe.")

@st.cache_data(ttl=900, show_spinner=False)
def load_universe():
    df = pd.read_csv("tickers.csv")
    df["Ticker"] = df["Ticker"].astype(str).str.strip()
    return df

@st.cache_data(ttl=900, show_spinner=False)
def fetch_one(ticker):
    # ~18 months gives enough history for EMA200 plus comparison periods.
    hist = yf.download(ticker, period="18mo", interval="1d", auto_adjust=False,
                       progress=False, threads=False)
    if hist is None or hist.empty:
        return None, None

    # yfinance can return MultiIndex columns even for one ticker.
    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = hist.columns.get_level_values(0)

    hist = hist.dropna(subset=["Close"]).copy()
    if len(hist) < 220:
        return None, hist

    close = hist["Close"].astype(float)
    high = hist["High"].astype(float)
    vol = hist["Volume"].astype(float)

    hist["EMA20"] = close.ewm(span=20, adjust=False).mean()
    hist["EMA50"] = close.ewm(span=50, adjust=False).mean()
    hist["EMA150"] = close.ewm(span=150, adjust=False).mean()
    hist["EMA200"] = close.ewm(span=200, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    hist["RSI14"] = 100 - (100 / (1 + rs))

    hist["AvgVol20"] = vol.rolling(20).mean()
    hist["RelVol"] = vol / hist["AvgVol20"]
    hist["AvgValue20Cr"] = ((close * vol).rolling(20).mean()) / 1e7

    row = hist.iloc[-1]
    close_now = float(row["Close"])
    ema20 = float(row["EMA20"])
    ema50 = float(row["EMA50"])
    ema150 = float(row["EMA150"])
    ema200 = float(row["EMA200"])
    ema200_20ago = float(hist["EMA200"].iloc[-21]) if len(hist) >= 221 else np.nan
    high52 = float(high.tail(252).max())
    low52 = float(hist["Low"].astype(float).tail(252).min())
    dist_high = (close_now / high52 - 1) * 100
    relvol = float(row["RelVol"]) if pd.notna(row["RelVol"]) else np.nan
    rsi = float(row["RSI14"]) if pd.notna(row["RSI14"]) else np.nan
    avg_value = float(row["AvgValue20Cr"]) if pd.notna(row["AvgValue20Cr"]) else np.nan

    # Minervini-style trend template, expressed with EMAs.
    stage2 = (
        close_now > ema50 and
        ema50 > ema150 and
        ema150 > ema200 and
        ema200 > ema200_20ago and
        close_now >= 1.30 * low52 and
        close_now >= 0.75 * high52
    )

    ema_stack = ema20 > ema50 > ema150 > ema200
    within2 = dist_high >= -2
    within5 = dist_high >= -5
    fresh_high = close_now >= high52 * 0.999
    volume_breakout = (relvol >= 1.5) if pd.notna(relvol) else False

    score = 0
    score += 30 if stage2 else 0
    score += 20 if ema_stack else 0
    score += 20 if within2 else (10 if within5 else 0)
    score += 15 if volume_breakout else 0
    score += 10 if (pd.notna(rsi) and 60 <= rsi <= 75) else (5 if pd.notna(rsi) and 55 <= rsi < 80 else 0)
    score += 5 if (pd.notna(avg_value) and avg_value >= 50) else 0

    if stage2 and within2 and volume_breakout and ema_stack:
        action = "BUY"
    elif stage2 and within5:
        action = "WATCH"
    else:
        action = "AVOID"

    out = {
        "Ticker": ticker,
        "Price": close_now,
        "EMA20": ema20,
        "EMA50": ema50,
        "EMA150": ema150,
        "EMA200": ema200,
        "EMA Stack": ema_stack,
        "EMA200 Rising": ema200 > ema200_20ago,
        "52W High": high52,
        "% From 52W High": dist_high,
        "RSI14": rsi,
        "Rel Volume": relvol,
        "Avg Value ₹Cr": avg_value,
        "Stage 2": stage2,
        "Within 2%": within2,
        "Within 5%": within5,
        "Fresh 52W High": fresh_high,
        "Volume Breakout": volume_breakout,
        "Momentum Score": score,
        "Action": action,
    }
    return out, hist

universe = load_universe()

st.info("Tip: start with the included watchlist, then replace tickers.csv with your Nifty 500 universe.")

with st.spinner("Fetching market data and calculating momentum signals..."):
    records = []
    histories = {}
    for _, u in universe.iterrows():
        ticker = u["Ticker"]
        try:
            rec, hist = fetch_one(ticker)
            if rec:
                rec["Company"] = u.get("Company", ticker)
                rec["Sector"] = u.get("Sector", "")
                rec["Large Cap"] = u.get("Large Cap", "")
                records.append(rec)
                histories[ticker] = hist
        except Exception:
            pass

if not records:
    st.error("No market data could be loaded. Check your internet connection and ticker symbols.")
    st.stop()

df = pd.DataFrame(records)
df = df[(df["Price"] >= min_price) & (df["Avg Value ₹Cr"].fillna(0) >= min_avg_value_cr)]

if only_stage2:
    df = df[df["Stage 2"]]
if only_near_high:
    df = df[df["Within 5%"]]

df = df.sort_values(["Momentum Score", "% From 52W High"], ascending=[False, False])

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Stocks scanned", len(records))
c2.metric("Stage 2", int(pd.DataFrame(records)["Stage 2"].sum()))
c3.metric("Within 2% of high", int(pd.DataFrame(records)["Within 2%"].sum()))
c4.metric("Fresh 52W highs", int(pd.DataFrame(records)["Fresh 52W High"].sum()))
c5.metric("BUY signals", int((pd.DataFrame(records)["Action"] == "BUY").sum()))

st.subheader("🔥 Ranked Momentum Watchlist")

display_cols = [
    "Ticker","Company","Sector","Price","Momentum Score","Action","Stage 2","EMA Stack",
    "% From 52W High","RSI14","Rel Volume","Avg Value ₹Cr","Fresh 52W High"
]
show = df[display_cols].copy()
for col in ["Price","% From 52W High","RSI14","Rel Volume","Avg Value ₹Cr"]:
    show[col] = pd.to_numeric(show[col], errors="coerce").round(2)

st.dataframe(
    show,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Momentum Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100),
        "Stage 2": st.column_config.CheckboxColumn("Stage 2"),
        "EMA Stack": st.column_config.CheckboxColumn("20>50>150>200"),
        "Fresh 52W High": st.column_config.CheckboxColumn("Fresh High"),
        "% From 52W High": st.column_config.NumberColumn("% from High", format="%.2f%%"),
        "Price": st.column_config.NumberColumn("Price", format="₹%.2f"),
    }
)

st.download_button(
    "Download current ranking (CSV)",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="momentum_ranking.csv",
    mime="text/csv"
)

st.subheader("📊 Stock Detail")
choices = df["Ticker"].tolist()
if choices:
    selected = st.selectbox("Select stock", choices)
    srow = df[df["Ticker"] == selected].iloc[0]
    a,b,c,d = st.columns(4)
    a.metric("Price", f"₹{srow['Price']:.2f}")
    b.metric("Momentum Score", f"{int(srow['Momentum Score'])}/100")
    c.metric("% from 52W high", f"{srow['% From 52W High']:.2f}%")
    d.metric("RSI(14)", f"{srow['RSI14']:.1f}")

    hist = histories[selected].tail(180).copy()
    chart = hist[["Close","EMA20","EMA50","EMA200"]].copy()
    st.line_chart(chart, use_container_width=True)

    st.write(
        f"**Signal:** {srow['Action']}  |  "
        f"**Stage 2:** {'Yes' if srow['Stage 2'] else 'No'}  |  "
        f"**Relative Volume:** {srow['Rel Volume']:.2f}×"
    )

st.caption(
    "Educational screening tool only. Signals are mechanical and can produce false breakouts. "
    "Use position sizing and a defined stop-loss."
)
