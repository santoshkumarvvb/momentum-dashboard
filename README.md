
# Stage 2 Momentum Web Dashboard

A Streamlit dashboard for Indian momentum trading using Yahoo Finance.

## What it calculates
- EMA 20 / 50 / 150 / 200
- EMA 200 rising vs 20 sessions ago
- RSI(14)
- 52-week high distance
- Relative volume vs 20-day average
- Average traded value
- Stage 2 trend template
- Momentum score 0–100
- BUY / WATCH / AVOID

## Run on your computer

1. Install Python 3.11+
2. Open Terminal / PowerShell in this folder
3. Create an environment (recommended)
4. Install dependencies:

   pip install -r requirements.txt

5. Start the dashboard:

   streamlit run app.py

6. Your browser will open automatically.

## Make it a true web app

### Streamlit Community Cloud
- Put these files in a GitHub repository.
- Go to Streamlit Community Cloud.
- Deploy `app.py`.

### Other hosting options
It can also run on Render, Railway, a VPS, or a home server.

## Updating the stock universe
Edit `tickers.csv`. NSE symbols must use `.NS`, e.g. `BAJFINANCE.NS`.

You can replace the starter list with the full Nifty 500.

## Automatic refresh
The app auto-refreshes at the interval selected in the sidebar.
Yahoo Finance calls are cached for 15 minutes.

## Stage 2 definition used
- Close > EMA50
- EMA50 > EMA150
- EMA150 > EMA200
- EMA200 > EMA200 from 20 sessions ago
- Close at least 30% above the 52-week low
- Close within 25% of the 52-week high

The dashboard separately flags:
- EMA20 > EMA50 > EMA150 > EMA200
- Within 2% of 52W high
- Fresh 52W high
- Relative volume >= 1.5

BUY is only produced when:
- Stage 2 is true
- EMA stack is bullish
- Price is within 2% of the 52-week high
- Relative volume >= 1.5

This is a mechanical screening rule, not investment advice.
