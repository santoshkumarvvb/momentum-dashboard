
import io
import requests
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="Stage 2 Momentum Dashboard", page_icon="📈", layout="wide")

st.title("📈 Stage 2 Momentum Dashboard")
st.caption("India: Nifty 500 auto-universe • Prices/history: Yahoo Finance")

NIFTY500_URL = "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv"

with st.sidebar:
    st.header("Settings")
    refresh_minutes = st.selectbox("Auto refresh", [5, 10, 15, 30, 60], index=2)
    st_autorefresh(interval=refresh_minutes * 60 * 1000, key="market_refresh")

    only_stage2 = st.checkbox("Stage 2 only", value=False)
    only_25 = st.checkbox("Within 25% of 52W high", value=False)
    only_5 = st.checkbox("Within 5% of 52W high", value=False)
    only_2 = st.checkbox("Within 2% of 52W high", value=False)

@st.cache_data(ttl=86400, show_spinner=False)
def load_india():
    """
    Load current Nifty 500 constituents from NSE Indices.
    Converts NSE symbols automatically to Yahoo Finance .NS tickers.
    Falls back to tickers.csv if NSE download is temporarily unavailable.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "text/csv,text/plain,*/*",
        }
        response = requests.get(NIFTY500_URL, headers=headers, timeout=20)
        response.raise_for_status()

        raw = pd.read_csv(io.BytesIO(response.content))

        symbol_col = next(
            c for c in raw.columns
            if str(c).strip().lower() == "symbol"
        )

        company_col = next(
            (c for c in raw.columns if "company" in str(c).lower()),
            symbol_col
        )

        sector_col = next(
            (
                c for c in raw.columns
                if "industry" in str(c).lower()
                or "sector" in str(c).lower()
            ),
            None
        )

        df = pd.DataFrame({
            "Ticker": raw[symbol_col].astype(str).str.strip() + ".NS",
            "Company": raw[company_col].astype(str),
            "Sector": raw[sector_col].astype(str) if sector_col is not None else "",
        })

        return df.drop_duplicates(subset=["Ticker"]).reset_index(drop=True)

    except Exception:
        # Backup only. Keep your current tickers.csv in GitHub.
        backup = pd.read_csv("tickers.csv")
        return backup[["Ticker", "Company", "Sector"]]

@st.cache_data(ttl=900, show_spinner=False)
def fetch_one(ticker):
    hist = yf.download(
        ticker,
        period="18mo",
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )

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
    volume = hist["Volume"].astype(float)

    for n in [20, 50, 150, 200]:
        hist[f"EMA{n}"] = close.ewm(span=n, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    hist["RSI14"] = 100 - (100 / (1 + rs))

    hist["AvgVol20"] = volume.rolling(20).mean()
    hist["RelVol"] = volume / hist["AvgVol20"]

    row = hist.iloc[-1]

    price = float(row["Close"])
    ema20 = float(row["EMA20"])
    ema50 = float(row["EMA50"])
    ema150 = float(row["EMA150"])
    ema200 = float(row["EMA200"])
    ema200_20ago = float(hist["EMA200"].iloc[-21])

    high52 = float(high.tail(252).max())
    low52 = float(low.tail(252).min())

    dist_high = (price / high52 - 1) * 100

    rsi = float(row["RSI14"]) if pd.notna(row["RSI14"]) else np.nan
    relvol = float(row["RelVol"]) if pd.notna(row["RelVol"]) else np.nan

    stage2 = (
        price > ema50
        and ema50 > ema150
        and ema150 > ema200
        and ema200 > ema200_20ago
        and price >= 1.30 * low52
        and price >= 0.75 * high52
    )

    ema_stack = ema20 > ema50 > ema150 > ema200
    within25 = dist_high >= -25
    within5 = dist_high >= -5
    within2 = dist_high >= -2
    fresh_high = price >= high52 * 0.999
    volume_breakout = bool(pd.notna(relvol) and relvol >= 1.5)

    score = 0
    score += 30 if stage2 else 0
    score += 20 if ema_stack else 0
    score += 20 if within2 else (10 if within5 else (5 if within25 else 0))
    score += 15 if volume_breakout else 0
    score += 10 if pd.notna(rsi) and 60 <= rsi <= 75 else (
        5 if pd.notna(rsi) and 55 <= rsi < 80 else 0
    )

    if stage2 and ema_stack and within2 and volume_breakout:
        action = "BUY"
    elif stage2 and within5:
        action = "WATCH"
    elif stage2 and within25:
        action = "STAGE 2"
    else:
        action = "AVOID"

    out = {
        "Ticker": ticker,
        "Price": price,
        "EMA20": ema20,
        "EMA50": ema50,
        "EMA150": ema150,
        "EMA200": ema200,
        "EMA Stack": ema_stack,
        "EMA200 Rising": ema200 > ema200_20ago,
        "52W High": high52,
        "% From 52W High": dist_high,
        "Within 25%": within25,
        "Within 5%": within5,
        "Within 2%": within2,
        "Fresh 52W High": fresh_high,
        "RSI14": rsi,
        "Rel Volume": relvol,
        "Volume Breakout": volume_breakout,
        "Stage 2": stage2,
        "Momentum Score": score,
        "Action": action,
    }

    return out, hist

universe = load_india()

st.info(
    f"Loaded {len(universe)} Nifty 500 symbols. "
    "Indian price/history data is fetched from Yahoo Finance using .NS tickers."
)

records = []
histories = {}

with st.spinner("Downloading Yahoo Finance data and calculating momentum signals..."):
    progress = st.progress(0)
    total = max(len(universe), 1)

    for i, (_, stock) in enumerate(universe.iterrows(), start=1):
        ticker = str(stock["Ticker"]).strip()

        try:
            rec, hist = fetch_one(ticker)

            if rec is not None:
                rec["Company"] = stock.get("Company", ticker)
                rec["Sector"] = stock.get("Sector", "")
                records.append(rec)
                histories[ticker] = hist

        except Exception:
            pass

        progress.progress(i / total)

    progress.empty()

if not records:
    st.error("No usable Yahoo Finance data was returned. Please try again later.")
    st.stop()

df = pd.DataFrame(records)
all_df = df.copy()

if only_stage2:
    df = df[df["Stage 2"]]

if only_25:
    df = df[df["Within 25%"]]

if only_5:
    df = df[df["Within 5%"]]

if only_2:
    df = df[df["Within 2%"]]

df = df.sort_values(
    ["Momentum Score", "% From 52W High"],
    ascending=[False, False]
).reset_index(drop=True)

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Stocks scanned", len(all_df))
c2.metric("Stage 2", int(all_df["Stage 2"].sum()))
c3.metric("Within 25%", int(all_df["Within 25%"].sum()))
c4.metric("Within 2%", int(all_df["Within 2%"].sum()))
c5.metric("BUY signals", int((all_df["Action"] == "BUY").sum()))

st.subheader("🔥 Nifty 500 Momentum Ranking")

display_cols = [
    "Ticker",
    "Company",
    "Sector",
    "Price",
    "Momentum Score",
    "Action",
    "Stage 2",
    "EMA20",
    "EMA50",
    "EMA150",
    "EMA200",
    "% From 52W High",
    "RSI14",
    "Rel Volume",
    "Fresh 52W High",
]

show = df[display_cols].copy()

for col in [
    "Price",
    "EMA20",
    "EMA50",
    "EMA150",
    "EMA200",
    "% From 52W High",
    "RSI14",
    "Rel Volume",
]:
    show[col] = pd.to_numeric(show[col], errors="coerce").round(2)

st.dataframe(show, use_container_width=True, hide_index=True)

st.download_button(
    "Download current ranking",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="nifty500_momentum_ranking.csv",
    mime="text/csv",
)

if not df.empty:
    st.subheader("📊 Stock Detail")

    selected = st.selectbox("Select stock", df["Ticker"].tolist())
    row = df[df["Ticker"] == selected].iloc[0]

    a, b, c, d = st.columns(4)

    a.metric("Price", f"₹{row['Price']:.2f}")
    b.metric("Momentum Score", f"{int(row['Momentum Score'])}/100")
    c.metric("% from 52W high", f"{row['% From 52W High']:.2f}%")
    d.metric("RSI(14)", f"{row['RSI14']:.1f}")

    chart = histories[selected].tail(180)[
        ["Close", "EMA20", "EMA50", "EMA200"]
    ]

    st.line_chart(chart, use_container_width=True)

    st.write(
        f"**Signal:** {row['Action']} | "
        f"**Stage 2:** {'Yes' if row['Stage 2'] else 'No'} | "
        f"**Relative volume:** {row['Rel Volume']:.2f}×"
    )

st.caption(
    "Nifty 500 membership is obtained from NSE Indices. "
    "Prices/history and technical calculations are based on Yahoo Finance data. "
    "Educational screening tool only."
)
