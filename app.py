
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="Global Stage 2 Momentum", page_icon="📈", layout="wide")
st.title("📈 Global Stage 2 Momentum Dashboard")
st.caption("Market prices/history: Yahoo Finance • Stage 2 / EMA / RSI / 52-week high")

with st.sidebar:
    st.header("Market")
    market = st.selectbox(
        "Universe",
        ["India Watchlist", "S&P 500", "Nasdaq-100", "USA Combined"]
    )
    refresh_minutes = st.selectbox("Auto refresh", [5, 10, 15, 30, 60], index=2)
    st_autorefresh(interval=refresh_minutes * 60 * 1000, key="refresh")

    st.header("Filters")
    only_stage2 = st.checkbox("Stage 2 only", False)
    only_25 = st.checkbox("Within 25% of 52W high", False)
    only_5 = st.checkbox("Within 5% of 52W high", False)
    only_2 = st.checkbox("Within 2% of 52W high", False)

@st.cache_data(ttl=86400)
def load_india():
    df = pd.read_csv("tickers.csv")
    keep = ["Ticker", "Company", "Sector"]
    for col in keep:
        if col not in df.columns:
            df[col] = ""
    return df[keep].dropna(subset=["Ticker"])

@st.cache_data(ttl=86400)
def load_sp500():
    raw = pd.read_csv("sp500.csv")
    return pd.DataFrame({
        "Ticker": raw["Symbol"].astype(str).str.replace(".", "-", regex=False),
        "Company": raw["Security"].astype(str),
        "Sector": raw["GICS Sector"].astype(str)
    })

@st.cache_data(ttl=86400)
def load_nasdaq100():
    raw = pd.read_csv("nasdaq100.csv")
    return pd.DataFrame({
        "Ticker": raw["Ticker"].astype(str).str.replace(".", "-", regex=False),
        "Company": raw["Company"].astype(str),
        "Sector": raw["GICS_Sector"].astype(str)
    })

def get_universe(name):
    if name == "India Watchlist":
        return load_india()
    if name == "S&P 500":
        return load_sp500()
    if name == "Nasdaq-100":
        return load_nasdaq100()
    return (
        pd.concat([load_sp500(), load_nasdaq100()], ignore_index=True)
        .drop_duplicates(subset=["Ticker"])
        .reset_index(drop=True)
    )

def calculate_one(hist, ticker):
    if hist is None or hist.empty:
        return None
    hist = hist.copy()
    needed = ["Close", "High", "Low", "Volume"]
    if not all(c in hist.columns for c in needed):
        return None
    hist = hist.dropna(subset=["Close"])
    if len(hist) < 220:
        return None

    close = pd.to_numeric(hist["Close"], errors="coerce")
    high = pd.to_numeric(hist["High"], errors="coerce")
    low = pd.to_numeric(hist["Low"], errors="coerce")
    volume = pd.to_numeric(hist["Volume"], errors="coerce")

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
    c = float(row["Close"])
    e20 = float(row["EMA20"])
    e50 = float(row["EMA50"])
    e150 = float(row["EMA150"])
    e200 = float(row["EMA200"])
    e200_20ago = float(hist["EMA200"].iloc[-21])

    h52 = float(high.tail(252).max())
    l52 = float(low.tail(252).min())
    dist_high = (c / h52 - 1) * 100

    rsi = float(row["RSI14"]) if pd.notna(row["RSI14"]) else np.nan
    relvol = float(row["RelVol"]) if pd.notna(row["RelVol"]) else np.nan

    # Stage 2 / Minervini-style trend template.
    stage2 = (
        c > e50
        and e50 > e150
        and e150 > e200
        and e200 > e200_20ago
        and c >= 1.30 * l52
        and c >= 0.75 * h52
    )

    ema_stack = e20 > e50 > e150 > e200
    within25 = dist_high >= -25
    within5 = dist_high >= -5
    within2 = dist_high >= -2
    fresh_high = c >= h52 * 0.999
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

    return {
        "Ticker": ticker,
        "Price": c,
        "EMA20": e20,
        "EMA50": e50,
        "EMA150": e150,
        "EMA200": e200,
        "EMA Stack": ema_stack,
        "EMA200 Rising": e200 > e200_20ago,
        "52W High": h52,
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
    }, hist

@st.cache_data(ttl=900, show_spinner=False)
def yahoo_batch_download(tickers_tuple):
    tickers = list(tickers_tuple)
    if not tickers:
        return {}
    result = {}
    batch_size = 40
    for start in range(0, len(tickers), batch_size):
        batch = tickers[start:start + batch_size]
        try:
            data = yf.download(
                tickers=batch,
                period="18mo",
                interval="1d",
                auto_adjust=False,
                group_by="ticker",
                threads=True,
                progress=False,
            )
        except Exception:
            continue

        if data is None or data.empty:
            continue

        if len(batch) == 1:
            result[batch[0]] = data
            continue

        # group_by="ticker" normally gives ticker as the first MultiIndex level.
        if isinstance(data.columns, pd.MultiIndex):
            level0 = set(map(str, data.columns.get_level_values(0)))
            for ticker in batch:
                try:
                    if ticker in level0:
                        sub = data[ticker].copy()
                    else:
                        # Fallback for reversed MultiIndex shape.
                        sub = data.xs(ticker, axis=1, level=1).copy()
                    result[ticker] = sub
                except Exception:
                    pass
    return result

universe = get_universe(market)
universe["Ticker"] = universe["Ticker"].astype(str).str.strip()
universe = universe[universe["Ticker"] != ""].drop_duplicates("Ticker").reset_index(drop=True)

st.write(f"**Universe:** {market} — {len(universe)} symbols")
st.info("All price, high, low and volume history used below is downloaded from Yahoo Finance.")

with st.spinner("Downloading Yahoo Finance history and calculating signals..."):
    history_map = yahoo_batch_download(tuple(universe["Ticker"].tolist()))
    records = []
    chart_history = {}
    for _, row in universe.iterrows():
        ticker = row["Ticker"]
        hist = history_map.get(ticker)
        calc = calculate_one(hist, ticker)
        if calc is None:
            continue
        rec, processed_hist = calc
        rec["Company"] = row.get("Company", "")
        rec["Sector"] = row.get("Sector", "")
        records.append(rec)
        chart_history[ticker] = processed_hist

if not records:
    st.error("Yahoo Finance returned no usable history. Try Refresh later.")
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
c1.metric("Scanned", len(all_df))
c2.metric("Stage 2", int(all_df["Stage 2"].sum()))
c3.metric("Within 25%", int(all_df["Within 25%"].sum()))
c4.metric("Within 2%", int(all_df["Within 2%"].sum()))
c5.metric("BUY", int((all_df["Action"] == "BUY").sum()))

st.subheader("🔥 Momentum Ranking")

cols = [
    "Ticker", "Company", "Sector", "Price", "Momentum Score", "Action",
    "Stage 2", "EMA20", "EMA50", "EMA150", "EMA200",
    "% From 52W High", "RSI14", "Rel Volume", "Fresh 52W High"
]
show = df[cols].copy()

for col in ["Price", "EMA20", "EMA50", "EMA150", "EMA200",
            "% From 52W High", "RSI14", "Rel Volume"]:
    show[col] = pd.to_numeric(show[col], errors="coerce").round(2)

st.dataframe(
    show,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Momentum Score": st.column_config.ProgressColumn(
            "Score", min_value=0, max_value=100
        ),
        "Stage 2": st.column_config.CheckboxColumn("Stage 2"),
        "Fresh 52W High": st.column_config.CheckboxColumn("Fresh 52W High"),
        "% From 52W High": st.column_config.NumberColumn(
            "% from 52W High", format="%.2f%%"
        ),
    },
)

st.download_button(
    "Download current ranking",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name=f"{market.replace(' ', '_')}_momentum.csv",
    mime="text/csv",
)

if not df.empty:
    st.subheader("📊 Stock Detail")
    selected = st.selectbox("Select stock", df["Ticker"].tolist())
    sr = df[df["Ticker"] == selected].iloc[0]

    a, b, c, d = st.columns(4)
    a.metric("Price", f"{sr['Price']:.2f}")
    b.metric("Momentum Score", f"{int(sr['Momentum Score'])}/100")
    c.metric("% from 52W high", f"{sr['% From 52W High']:.2f}%")
    d.metric("RSI(14)", f"{sr['RSI14']:.1f}")

    hist = chart_history[selected].tail(180)
    st.line_chart(hist[["Close", "EMA20", "EMA50", "EMA200"]], use_container_width=True)

    st.write(
        f"**Signal:** {sr['Action']} | "
        f"**Stage 2:** {'Yes' if sr['Stage 2'] else 'No'} | "
        f"**Relative volume:** {sr['Rel Volume']:.2f}×"
    )

st.caption(
    "Index CSV files define which symbols are scanned. Market prices/history and "
    "all technical calculations are based on Yahoo Finance data. "
    "Educational screening tool only; not investment advice."
)
