# PortfolioLab

PortfolioLab is a quantitative research console for building stock universes, testing portfolio/signal ideas, and reviewing backtest behavior visually.

The current app is centered on the interactive `PortfolioLab Console`: a local web UI backed by a small Python standard-library server. It uses Yahoo Finance data through `yfinance` when available, caches downloaded prices locally, and lets you save/edit strategy and neutralization code from the browser.

<div align="center">
  <video src="https://github.com/user-attachments/assets/e7fed93f-606d-4f9a-9f1d-8299b831d95f" width="850" autoplay loop muted playsinline></video>
</div>

## Current Features

- Search and add the Yahoo Finance symbols you are interested in.
- Refresh daily OHLCV data only for the current Available Stocks set, with a local CSV cache.
- Inspect raw stock prices and candlesticks in Stock Research.
- Choose which research symbols enter the portfolio; unchecked portfolio names keep weight fixed at zero.
- Preprocess market data before signal construction, including none, backfill, forward-fill, or custom Python code.
- Use equal weights, manual weights, built-in momentum/reversion strategies, or custom Python strategy code.
- Save, update, and delete strategy snippets in `strategies/strategy_library.json`.
- Apply no, market, region, sector, industry, or custom neutralization code.
- Backtest with transaction costs, slippage, train/test split, and long/short weights.
- Review PnL, normalized prices, signal weights, drawdown, Sharpe, and turnover plots.
- Review aggregate/training/testing metrics, yearly metrics, and selected-stock exposure tables.
- Use the notebook workflow for code-first experiments without the UI.

## Quick Start

PortfolioLab runs on Python 3.10 or newer. From the project folder, either use your existing local venv:

```bash
source portfoliolab_venv/bin/activate
PYTHONPATH=src python3 -m portfoliolab.app
```

For a fresh clone, install the app and the research/data extras first:

```bash
python3 -m venv portfoliolab_venv
source portfoliolab_venv/bin/activate
python3 -m pip install -e ".[market-data,research,notebook,test]"
python3 -m portfoliolab.app
```

Then open:

```text
http://127.0.0.1:8765
```

If you install the project later, the script entry point also launches the same console:

```bash
portfoliolab
```

## Market Data

PortfolioLab separates symbol lookup from price history:

- `Universe` lets you type ticker symbols such as `AAPL, MSFT, BRK-B, MC.PA` and checks whether Yahoo Finance can return them.
- `Available Stocks` keeps the symbols you manually added. Its checkbox controls whether the symbol appears in `Stock Research`, and its delete button removes the symbol from the active universe.
- The `Portfolio` checkbox inside `Stock Research` controls whether that symbol enters the backtest. If it is not checked for the portfolio, its weight is fixed at `0`.
- `Refresh Price Data` downloads OHLCV history for the symbols still in `Available Stocks` and rewrites the local cache without symbols you removed.
- The displayed region/country is intended to describe the issuer, not only the exchange listing venue. Old Nasdaq catalog cache rows can label ADRs as U.S.-listed; manual Yahoo lookup and curated seed metadata override that for known symbols.

The price cache lives at:

```text
data/yahoo_prices.csv
```

The optional lightweight symbol catalog cache lives at:

```text
data/catalogs/yahoo_us_catalog.json
```

The app uses Yahoo Finance / `yfinance` to validate tickers and fetch selected price histories. Install `yfinance` in your environment if needed:

```bash
python3 -m pip install -e ".[market-data]"
```

Generated price/catalog cache files are intentionally ignored by git. Keep `data/.gitkeep` so the cache folder exists in a fresh checkout, but do not commit large downloaded market data unless you deliberately want a reproducible sample snapshot.

## Preprocessing Code

Preprocessors are edited in the `Data Preprocessing` panel. A preprocessor should expose:

```python
def preprocess(data, symbols, context):
    return data
```

The function receives the loaded `MarketData`, selected portfolio symbols, and a small context dictionary. It must return a `MarketData` object. The editor exposes `pd`, `np`, `Bar`, and `MarketData`, so custom code can use pandas/numpy or create filled bars directly.

For nontrivial cleaning, use the DataFrame helpers. The built-in backfill and forward-fill examples use `pandas`:

```python
def preprocess(data, symbols, context):
    frame = data.to_frame()
    # edit frame with pandas/numpy
    return MarketData.from_frame(frame)
```

Install the research extras, or install `pandas` and `numpy` directly, before using DataFrame preprocessing:

```bash
python3 -m pip install pandas numpy
```

Saved preprocessors live in:

```text
strategies/preprocessing_library.json
```

The file is initialized with the built-in examples and is updated when you save custom preprocessing code from the UI.

## Strategy Code

Strategies are edited in the `Strategy Construction` panel. A strategy should define a `Strategy` class like:

```python
class Strategy:
    def compute(self, data, as_of, context):
        return dict(BASE_WEIGHTS)
```

The returned dictionary is the target portfolio weight after the close on `as_of`, held until the next trading date or rebalance. Strategy code can use `SELECTED_SYMBOLS`, `BASE_WEIGHTS`, `pd`, `np`, `Bar`, and `MarketData`. For table-style logic, call `data.to_frame()` to get columns `date`, `symbol`, `open`, `high`, `low`, `close`, and `volume`.

Saved strategies live in:

```text
strategies/strategy_library.json
```

Neutralization snippets live in:

```text
strategies/neutralization_library.json
```

Neutralization code can define both pieces:

```python
def group_for(symbol, stock):
    # stock is the metadata row from Available Stocks.
    # Common fields: symbol, name, region, sector, industry, exchange.
    region = stock.get("region") or "Other"
    sector = stock.get("sector") or "Other"

    if region in {"China", "Japan", "Taiwan"}:
        return "Asia"
    if sector in {"Technology", "Communication"}:
        return "Growth"
    return region

def neutralize(alpha, groups):
    ...
```

`group_for` is optional. When provided, it receives each selected symbol and its metadata, then returns the group label used by `neutralize(alpha, groups)`.
Neutralization code also has access to `pd`, `np`, `Bar`, and `MarketData` for custom grouping helpers.

## Metrics Notes

PortfolioLab follows the current research definitions used in the console:

- `PnL`: daily profit and loss in dollars.
- `Returns`: annualized PnL divided by half book size.
- `Sharpe`: annualized information ratio using daily returns.
- `Turnover`: dollars traded divided by book size.
- `Margin`: PnL divided by dollars traded.
- `Drawdown`: peak-to-trough PnL drawdown divided by half book size.
- `Fitness`: `Sharpe * sqrt(abs(Returns) / max(Turnover, 0.125))`.

The `Selected Stocks` table describes raw instrument behavior and exposure. Its stock `Return` is raw price return over the backtest window, while the `Performance` returns are portfolio/strategy results after weights, long/short direction, costs, and rebalancing.

## Tests

Run the test suite with:

```bash
PYTHONPATH=src python3 -m pytest -q
```

The tests are also compatible with `python3 -m unittest discover -s tests`.

## Project Layout

```text
src/portfoliolab/app.py         Local web console and HTTP API
src/portfoliolab/backtest.py    Backtest engine
src/portfoliolab/metrics.py     Performance metrics
src/portfoliolab/strategies.py  Built-in and custom strategy helpers
src/portfoliolab/providers.py   Yahoo Finance data refresh/cache helpers
notebooks/                     Code-first research notebooks
strategies/                    Saved strategy and neutralization snippets
data/                          Local market-data cache; generated files are git-ignored
tests/                         Regression tests
```

The main notebook is:

```text
notebooks/PortfolioLab_research_workflow.ipynb
```

It is a self-contained, code-friendly version of the workflow from data loading through backtesting and plotting.

## Roadmap

- Better universe metadata, including fuller sector and industry mappings.
- More data-provider adapters.
- Experiment persistence and comparison.
- Richer strategy diagnostics and attribution.
- True out-of-sample / live-paper-trading workflow once the research layer is stable.
