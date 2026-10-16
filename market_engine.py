"""Shared indicator and universe calculations for the UI and background worker."""

import io
import requests
import pandas as pd
import numpy as np
import yfinance as yf

NIFTY500_URL = 'https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv'

EUROPE = [('AZN.L', 'AstraZeneca', 'UK', 'GBP', 'Health Care'), ('SHEL.L', 'Shell', 'UK', 'GBP', 'Energy'), ('HSBA.L', 'HSBC Holdings', 'UK', 'GBP', 'Financials'), ('ULVR.L', 'Unilever', 'UK', 'GBP', 'Consumer Staples'), ('REL.L', 'RELX', 'UK', 'GBP', 'Industrials'), ('LSEG.L', 'London Stock Exchange Group', 'UK', 'GBP', 'Financials'), ('RR.L', 'Rolls-Royce Holdings', 'UK', 'GBP', 'Industrials'), ('BA.L', 'BAE Systems', 'UK', 'GBP', 'Industrials'), ('NG.L', 'National Grid', 'UK', 'GBP', 'Utilities'), ('GSK.L', 'GSK', 'UK', 'GBP', 'Health Care'), ('SAP.DE', 'SAP', 'Germany', 'EUR', 'Technology'), ('SIE.DE', 'Siemens', 'Germany', 'EUR', 'Industrials'), ('ALV.DE', 'Allianz', 'Germany', 'EUR', 'Financials'), ('DTE.DE', 'Deutsche Telekom', 'Germany', 'EUR', 'Communication Services'), ('MUV2.DE', 'Munich Re', 'Germany', 'EUR', 'Financials'), ('RHM.DE', 'Rheinmetall', 'Germany', 'EUR', 'Industrials'), ('AIR.PA', 'Airbus', 'France', 'EUR', 'Industrials'), ('MC.PA', 'LVMH', 'France', 'EUR', 'Consumer Discretionary'), ('OR.PA', "L'Oreal", 'France', 'EUR', 'Consumer Staples'), ('SU.PA', 'Schneider Electric', 'France', 'EUR', 'Industrials'), ('SAF.PA', 'Safran', 'France', 'EUR', 'Industrials'), ('TTE.PA', 'TotalEnergies', 'France', 'EUR', 'Energy'), ('ASML.AS', 'ASML Holding', 'Netherlands', 'EUR', 'Technology'), ('INGA.AS', 'ING Groep', 'Netherlands', 'EUR', 'Financials'), ('PHIA.AS', 'Philips', 'Netherlands', 'EUR', 'Health Care'), ('AD.AS', 'Ahold Delhaize', 'Netherlands', 'EUR', 'Consumer Staples'), ('NOVN.SW', 'Novartis', 'Switzerland', 'CHF', 'Health Care'), ('ROG.SW', 'Roche', 'Switzerland', 'CHF', 'Health Care'), ('NESN.SW', 'Nestle', 'Switzerland', 'CHF', 'Consumer Staples'), ('UBSG.SW', 'UBS Group', 'Switzerland', 'CHF', 'Financials'), ('ABBN.SW', 'ABB', 'Switzerland', 'CHF', 'Industrials')]

def normalize_nse_list(raw, segment):
    raw = raw.rename(columns=lambda c: str(c).strip().upper().replace(' ', '_'))
    if 'SYMBOL' not in raw or 'NAME_OF_COMPANY' not in raw:
        raise ValueError('NSE security list is missing symbol/company columns')
    symbols = raw['SYMBOL'].fillna('').astype(str).str.strip()
    out = pd.DataFrame({'Ticker': symbols + '.NS', 'Company': raw['NAME_OF_COMPANY'].fillna('').astype(str).str.strip(), 'Sector': '', 'Country': 'India', 'Currency': 'INR', 'Segment': segment})
    return out[symbols.ne('')].drop_duplicates('Ticker')

def load_all_nse():
    sources = [('Mainboard', 'https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv'), ('SME', 'https://nsearchives.nseindia.com/emerge/corporates/content/SME_EQUITY_L.csv')]
    frames = []
    for segment, url in sources:
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0', 'Accept': 'text/csv,*/*'}, timeout=30)
        r.raise_for_status()
        frames.append(normalize_nse_list(pd.read_csv(io.BytesIO(r.content)), segment))
    out = pd.concat(frames, ignore_index=True).drop_duplicates('Ticker').reset_index(drop=True)
    if out.empty:
        raise ValueError('NSE security lists were empty')
    return out

def load_india():
    try:
        r = requests.get(NIFTY500_URL, headers={'User-Agent': 'Mozilla/5.0', 'Accept': 'text/csv,text/plain,*/*'}, timeout=20)
        r.raise_for_status()
        raw = pd.read_csv(io.BytesIO(r.content))
        sc = next((c for c in raw.columns if str(c).strip().lower() == 'symbol'))
        cc = next((c for c in raw.columns if 'company' in str(c).lower()), sc)
        sec = next((c for c in raw.columns if 'industry' in str(c).lower() or 'sector' in str(c).lower()), None)
        return pd.DataFrame({'Ticker': raw[sc].astype(str).str.strip() + '.NS', 'Company': raw[cc].astype(str), 'Sector': raw[sec].astype(str) if sec else '', 'Country': 'India', 'Currency': 'INR'}).drop_duplicates('Ticker')
    except Exception:
        raw = pd.read_csv('tickers.csv')
        out = raw[['Ticker', 'Company', 'Sector']].copy()
        out['Country'] = 'India'
        out['Currency'] = 'INR'
        return out

def load_sp500():
    r = pd.read_csv('sp500.csv')
    o = pd.DataFrame({'Ticker': r['Symbol'].astype(str).str.replace('.', '-', regex=False), 'Company': r['Security'], 'Sector': r['GICS Sector']})
    o['Country'] = 'USA'
    o['Currency'] = 'USD'
    return o.drop_duplicates('Ticker')

def load_nasdaq():
    r = pd.read_csv('nasdaq100.csv')
    o = pd.DataFrame({'Ticker': r['Ticker'].astype(str), 'Company': r['Company'], 'Sector': r.get('GICS_Sector', '')})
    o['Country'] = 'USA'
    o['Currency'] = 'USD'
    return o.drop_duplicates('Ticker')

def load_europe():
    return pd.DataFrame(EUROPE, columns=['Ticker', 'Company', 'Country', 'Currency', 'Sector'])

def get_universe(n):
    if n == 'India — All NSE stocks':
        return load_all_nse()
    if n == 'India — Nifty 500':
        return load_india()
    if n == 'USA — S&P 500':
        return load_sp500()
    if n == 'USA — Nasdaq 100':
        return load_nasdaq()
    if n == 'USA — Combined':
        return pd.concat([load_sp500(), load_nasdaq()]).drop_duplicates('Ticker')
    if n == 'Europe / UK':
        return load_europe()
    return pd.concat([load_all_nse(), load_sp500(), load_nasdaq(), load_europe()]).drop_duplicates('Ticker')

def benchmark_for(country):
    if country == 'India':
        return '^NSEI'
    if country == 'USA':
        return '^GSPC'
    if country == 'UK':
        return '^FTSE'
    if country == 'Germany':
        return '^GDAXI'
    if country == 'France':
        return '^FCHI'
    if country == 'Netherlands':
        return '^AEX'
    if country == 'Switzerland':
        return '^SSMI'
    return '^STOXX'

def period_return(s, days):
    s = s.dropna()
    if len(s) <= days:
        return np.nan
    return (float(s.iloc[-1]) / float(s.iloc[-days - 1]) - 1) * 100

def chartink_stage2(h):
    """User's cash Stage-2 template; Friday weeks include the current partial week."""
    close = h['Close'].astype(float)
    sma = {n: close.rolling(n).mean() for n in (50, 150, 200)}
    weekly = h[['High', 'Low']].resample('W-FRI').agg({'High': 'max', 'Low': 'min'}).dropna()
    weekly = weekly.tail(52)
    row = h.iloc[-1]
    p = float(row['Close'])
    s50, s150, s200 = (float(sma[n].iloc[-1]) for n in (50, 150, 200))
    rising = any((len(h) > offset and pd.notna(sma[200].iloc[-offset - 1]) and (s200 > float(sma[200].iloc[-offset - 1])) for offset in (20, 80, 100)))
    hi = float(weekly['High'].max())
    lo = float(weekly['Low'].min())
    avgvol50 = float(h['Volume'].astype(float).rolling(50).mean().iloc[-1])
    ready = len(weekly) == 52 and all((pd.notna(v) for v in (s50, s150, s200, avgvol50)))
    passed = bool(ready and p > s50 > s150 > s200 and rising and (float(row['Low']) >= 1.5 * lo) and (float(row['High']) >= 0.75 * hi) and (p > 10) and (p * avgvol50 > 1000000) and (float(row['Volume']) > 0))
    return {'Stage 2': passed, 'SMA50': s50, 'SMA150': s150, 'SMA200': s200, 'SMA200 Rising': bool(rising), 'Stage 2 Weekly High': hi, 'Stage 2 Weekly Low': lo, 'Avg Volume 50': avgvol50, 'Stage 2 History Ready': bool(ready)}

def recovery_ath_signals(h):
    """Daily model on all available, split-adjusted Yahoo history; not an order executor."""
    c = h['Close'].astype(float)
    ema = c.ewm(span=200, adjust=False, min_periods=200).mean()
    prior_high = h['High'].astype(float).cummax().shift(1)
    dates = pd.Series(h.index, index=h.index)
    last_below = dates.where(c < ema).ffill().shift(1)
    recent_flags = (last_below >= h.index - pd.DateOffset(months=6)).to_numpy()
    touches = (h['Low'].astype(float) <= ema).to_numpy()
    breakouts = ((h['High'].astype(float) > prior_high) & (c > ema)).to_numpy()
    active = False
    entry_date = ''
    signal = 'WAIT'
    for i in range(len(h)):
        signal = 'HOLD' if active else 'WAIT'
        if active and touches[i]:
            active = False
            signal = 'EXIT'
        elif not active and recent_flags[i] and breakouts[i]:
            if touches[i]:
                signal = 'AMBIGUOUS'
            else:
                active = True
                entry_date = str(h.index[i].date())
                signal = 'BUY'
    recent = bool(recent_flags[-1])
    return {'Recovery Signal': signal, 'Below EMA200 in 6M': bool(recent), 'Prior All-Time High': float(prior_high.iloc[-1]), 'Recovery EMA200': float(ema.iloc[-1]), 'Distance to EMA200 %': (float(c.iloc[-1]) / float(ema.iloc[-1]) - 1) * 100, 'Model Entry Date': entry_date if active or signal == 'EXIT' else '', 'History From': str(h.index[0].date()), 'Data Date': str(h.index[-1].date())}

def analyze_history(ticker, h, recovery_mode=False):
    if h is None or h.empty:
        return (None, None)
    if isinstance(h.columns, pd.MultiIndex):
        h.columns = h.columns.get_level_values(0)
    h = h.dropna(subset=['Close']).copy()
    if len(h) < 220:
        return (None, h)
    close, high, low, vol = [h[c].astype(float) for c in ['Close', 'High', 'Low', 'Volume']]
    for n in [20, 50, 100, 150, 200]:
        h[f'EMA{n}'] = close.ewm(span=n, adjust=False).mean()
    for n in [50, 150, 200]:
        h[f'SMA{n}'] = close.rolling(n).mean()
    d = close.diff()
    g = d.clip(lower=0)
    l = -d.clip(upper=0)
    ag = g.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    al = l.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = ag / al.replace(0, np.nan)
    h['RSI14'] = 100 - 100 / (1 + rs)
    h['AvgVol20'] = vol.rolling(20).mean()
    h['RelVol'] = vol / h['AvgVol20']
    row = h.iloc[-1]
    p = float(row['Close'])
    e20, e50, e100, e150, e200 = [float(row[f'EMA{n}']) for n in [20, 50, 100, 150, 200]]
    old = float(h['EMA200'].iloc[-21])
    hi = float(high.tail(252).max())
    lo = float(low.tail(252).min())
    dist = (p / hi - 1) * 100
    rsi = float(row['RSI14']) if pd.notna(row['RSI14']) else np.nan
    rv = float(row['RelVol']) if pd.notna(row['RelVol']) else np.nan
    r3, r6, r12 = (period_return(close, 63), period_return(close, 126), period_return(close, 252))
    stage_details = chartink_stage2(h)
    stage = stage_details['Stage 2']
    stack = e20 > e50 > e100 > e200
    w25, w10, w5, w2 = (dist >= -25, dist >= -10, dist >= -5, dist >= -2)
    fresh = p >= hi * 0.999
    vb = bool(pd.notna(rv) and rv >= 1.5)
    return ({**(recovery_ath_signals(h) if recovery_mode else {}), **stage_details, 'Ticker': ticker, 'Price': p, 'EMA20': e20, 'EMA50': e50, 'EMA100': e100, 'EMA150': e150, 'EMA200': e200, 'EMA Stack': stack, 'EMA200 Rising': e200 > old, '52W High': hi, '% From 52W High': dist, 'Within 25%': w25, 'Within 10%': w10, 'Within 5%': w5, 'Within 2%': w2, 'Fresh 52W High': fresh, 'RSI14': rsi, 'Rel Volume': rv, 'Volume Breakout': vb, 'Stage 2': stage, 'Return 3M %': r3, 'Return 6M %': r6, 'Return 12M %': r12}, h)

def extract_ticker_history(batch, ticker, single=False):
    if batch is None or batch.empty:
        return pd.DataFrame()
    if isinstance(batch.columns, pd.MultiIndex):
        for level in range(batch.columns.nlevels):
            if ticker in batch.columns.get_level_values(level):
                return batch.xs(ticker, axis=1, level=level).copy()
        return pd.DataFrame()
    return batch.copy() if single else pd.DataFrame()
