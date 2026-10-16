# Global momentum dashboard

Three strategies: Chartink SMA Stage 2, custom momentum ranking, and EMA200 recovery to an all-time-high breakout.

## Prepared daily results

The dashboard defaults to prepared snapshots from this repository's `market-data` release. This mode does not download individual stock histories. Snapshots survive browser reconnects and Streamlit host replacement.

The GitHub Actions background updater runs independently of the browser:

- India: weekdays at 12:47 UTC / 18:17 IST.
- USA and Europe: weekdays at 22:47 UTC / 04:17 IST the next day.
- Changes to the workflow, updater or shared engine trigger an initial build.
- Use Run workflow on GitHub Actions for a manual background update.

Scheduled runs can be delayed by GitHub. Snapshot timestamps, per-stock Data Date, and coverage are displayed. This is daily data, not live quotes.

The first run fetches full available Yahoo history. Later runs restore regional SQLite databases from release assets and download an overlapping recent window. Older prices are retained for ATH and EMA calculations. New stocks and recent split/bonus events trigger a full-history fetch for the affected ticker. No paid data account or separate database credentials are needed.

Assets: india.json.gz, usa.json.gz, europe.json.gz and prices-<region>.sqlite.gz. Only public market data and derived signals are published. If fewer than half of the universe can be updated or analysed, previous prepared results remain; successful database updates are retained for the next run.

## Strategies

Stage 2 follows the user's exact Chartink template: Close > SMA50 > SMA150 > SMA200; SMA200 above its value 20 OR 80 OR 100 sessions ago; daily low >= 1.5 times the weekly 52-week low; daily high >= 0.75 times the weekly 52-week high; close >10; close times average 50-day volume >1,000,000; volume >0. International thresholds use local quoted currency.

The custom momentum score combines trend, EMA alignment, high proximity, RSI, relative volume, 3/6/12-month return ranks and benchmark outperformance. It is not a published or backtested investment model.

EMA200 recovery: a daily close below EMA200 within the previous six calendar months, then a fresh intraday high above every prior high available in the data, with close above EMA200. Exit when a subsequent daily low touches EMA200. Same-bar breakout/EMA touch is AMBIGUOUS because daily bars do not establish event order. BUY/HOLD/EXIT describe simulated positions, not actual holdings or orders. Daily-bar exits are approximations, not guaranteed fills.

Yahoo history may be incomplete. ATH means highest in all available provider history. OHLC is provider split-adjusted and does not include dividend reinvestment. Unfinished trading sessions are excluded from prepared signals.

## Universes and manual mode

India includes official NSE mainboard plus SME lists, with Nifty 500 optional. USA: S&P 500, Nasdaq 100 or their union. Europe/UK retains the curated list. Global combines these regions.

Untick Use prepared daily results (fast) for manual scanning. Manual scans save batches on the hosting server and resume with Start / resume. Those local checkpoints can disappear when the server resets; published background snapshots remain available.

## Development

Use Python 3.11+, install requirements.txt, run `python -m unittest test_background -v`, then `streamlit run app.py`.

The worker uses GitHub CLI with release-write permission: `python background_update.py --region india`. It updates incrementally unless `--full` is supplied. The local .background-data directory is ignored by Git.
