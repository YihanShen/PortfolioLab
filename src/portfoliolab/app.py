from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from datetime import date
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from portfoliolab.backtest import BacktestConfig
from portfoliolab.data import MarketData, parse_date
from portfoliolab.evaluation import run_is_os_evaluation
from portfoliolab.metrics import evaluate
from portfoliolab.providers import (
    DataProviderError,
    fetch_nasdaq_us_symbol_catalog,
    fetch_yahoo_data,
    lookup_yahoo_symbols,
    write_market_data_csv,
)
from portfoliolab.strategies import (
    FixedWeightStrategy,
    NeutralizedStrategy,
    SelectedMeanReversionStrategy,
    SelectedMomentumStrategy,
    equal_weights,
    load_inline_neutralizer,
    load_inline_preprocessor,
    load_inline_strategy,
    normalize_weights,
)


@dataclass(frozen=True)
class StockInfo:
    symbol: str
    name: str
    region: str
    source: str
    sector: str
    exchange: str = ""
    provider_symbol: str | None = None
    industry: str = ""


STOCKS = [
    StockInfo("AAPL", "Apple", "United States", "Yahoo Finance", "Technology", "NASDAQ"),
    StockInfo("AMZN", "Amazon", "United States", "Yahoo Finance", "Consumer", "NASDAQ"),
    StockInfo("ASML", "ASML", "Netherlands", "Yahoo Finance", "Technology", "NASDAQ"),
    StockInfo("BABA", "Alibaba", "China", "Yahoo Finance", "Consumer", "NYSE"),
    StockInfo("BRK-B", "Berkshire Hathaway", "United States", "Yahoo Finance", "Financials", "NYSE"),
    StockInfo("GOOGL", "Alphabet", "United States", "Yahoo Finance", "Technology", "NASDAQ"),
    StockInfo("HD", "Home Depot", "United States", "Yahoo Finance", "Consumer", "NYSE"),
    StockInfo("JPM", "JPMorgan Chase", "United States", "Yahoo Finance", "Financials", "NYSE"),
    StockInfo("JNJ", "Johnson & Johnson", "United States", "Yahoo Finance", "Healthcare", "NYSE"),
    StockInfo("KO", "Coca-Cola", "United States", "Yahoo Finance", "Staples", "NYSE"),
    StockInfo("LLY", "Eli Lilly", "United States", "Yahoo Finance", "Healthcare", "NYSE"),
    StockInfo("LVMH", "LVMH", "France", "Yahoo Finance", "Consumer", "Euronext Paris", "MC.PA"),
    StockInfo("MA", "Mastercard", "United States", "Yahoo Finance", "Financials", "NYSE"),
    StockInfo("META", "Meta", "United States", "Yahoo Finance", "Technology", "NASDAQ"),
    StockInfo("MSFT", "Microsoft", "United States", "Yahoo Finance", "Technology", "NASDAQ"),
    StockInfo("NESN", "Nestle", "Switzerland", "Yahoo Finance", "Staples", "SIX", "NESN.SW"),
    StockInfo("NFLX", "Netflix", "United States", "Yahoo Finance", "Communication", "NASDAQ"),
    StockInfo("NOVO", "Novo Nordisk", "Denmark", "Yahoo Finance", "Healthcare", "NYSE", "NVO"),
    StockInfo("NVDA", "Nvidia", "United States", "Yahoo Finance", "Technology", "NASDAQ"),
    StockInfo("ORCL", "Oracle", "United States", "Yahoo Finance", "Technology", "NYSE"),
    StockInfo("PEP", "PepsiCo", "United States", "Yahoo Finance", "Staples", "NASDAQ"),
    StockInfo("PFE", "Pfizer", "United States", "Yahoo Finance", "Healthcare", "NYSE"),
    StockInfo("PG", "Procter & Gamble", "United States", "Yahoo Finance", "Staples", "NYSE"),
    StockInfo("SAP", "SAP", "Germany", "Yahoo Finance", "Technology", "NYSE"),
    StockInfo("SONY", "Sony", "Japan", "Yahoo Finance", "Technology", "NYSE"),
    StockInfo("TM", "Toyota", "Japan", "Yahoo Finance", "Industrial", "NYSE"),
    StockInfo("TSLA", "Tesla", "United States", "Yahoo Finance", "Consumer", "NASDAQ"),
    StockInfo("TSM", "Taiwan Semiconductor", "Taiwan", "Yahoo Finance", "Technology", "NYSE"),
    StockInfo("UNH", "UnitedHealth", "United States", "Yahoo Finance", "Healthcare", "NYSE"),
    StockInfo("V", "Visa", "United States", "Yahoo Finance", "Financials", "NYSE"),
    StockInfo("WMT", "Walmart", "United States", "Yahoo Finance", "Staples", "NYSE"),
    StockInfo("XOM", "Exxon Mobil", "United States", "Yahoo Finance", "Energy", "NYSE"),
]

POOLS = [
    {"id": "us_growth", "name": "US Growth", "symbols": ["AAPL", "AMZN", "META", "MSFT", "NVDA"]},
    {"id": "us_defensive", "name": "US Defensive", "symbols": ["JNJ", "KO", "PG", "WMT"]},
    {"id": "europe", "name": "Europe", "symbols": ["ASML", "LVMH", "NESN", "NOVO", "SAP"]},
    {"id": "asia", "name": "Asia", "symbols": ["BABA", "SONY", "TM", "TSM"]},
    {"id": "global_quality", "name": "Global Quality", "symbols": ["AAPL", "ASML", "JNJ", "MSFT", "NESN", "NOVO", "TSM", "WMT"]},
]

YAHOO_SYMBOLS = {stock.symbol: stock.provider_symbol or stock.symbol for stock in STOCKS}

STOCK_INDUSTRIES = {
    "AAPL": "Technology Hardware",
    "AMZN": "Internet Retail",
    "ASML": "Semiconductors",
    "BABA": "Internet Retail",
    "JNJ": "Pharmaceuticals",
    "KO": "Beverages",
    "LVMH": "Luxury Goods",
    "META": "Interactive Media",
    "MSFT": "Software",
    "NESN": "Food Products",
    "NOVO": "Pharmaceuticals",
    "NVDA": "Semiconductors",
    "PG": "Household Products",
    "SAP": "Software",
    "SONY": "Consumer Electronics",
    "TM": "Automobiles",
    "TSM": "Semiconductors",
    "WMT": "Retail",
}

STOCK_SECTORS = {stock.symbol: stock.sector for stock in STOCKS if stock.sector}

DEFAULT_CODE = """class Strategy:
    def compute(self, data, as_of, context):
        # data is MarketData. For table work, use: frame = data.to_frame()
        # frame columns: date, symbol, open, high, low, close, volume
        # pd and np are available when pandas/numpy are installed.
        return dict(BASE_WEIGHTS)
"""

DEFAULT_MOMENTUM_CODE = """class Strategy:
    def compute(self, data, as_of, context):
        # Rank selected portfolio stocks by 6-month momentum,
        # skipping the most recent week to avoid very short-term reversal noise.
        scores = {}
        for symbol in SELECTED_SYMBOLS:
            score = data.return_over(symbol, as_of, lookback=126, skip_recent=5)
            if score is not None and score > 0:
                scores[symbol] = score

        selected = sorted(scores, key=scores.get, reverse=True)[:4]
        if not selected:
            return {}

        raw_alpha = {symbol: scores[symbol] for symbol in selected}
        gross = sum(abs(value) for value in raw_alpha.values())
        target_gross = min(context.max_gross_exposure, 1.0)
        return {symbol: value * target_gross / gross for symbol, value in raw_alpha.items()}
"""

DEFAULT_REVERSION_CODE = """class Strategy:
    def compute(self, data, as_of, context):
        # Rank selected portfolio stocks by 1-month return and buy the weakest names.
        # The returned values are raw alpha weights; neutralization may later turn
        # them into long/short relative positions inside each chosen group.
        scores = {}
        for symbol in SELECTED_SYMBOLS:
            score = data.return_over(symbol, as_of, lookback=21)
            if score is not None:
                scores[symbol] = score

        selected = sorted(scores, key=scores.get)[:4]
        if not selected:
            return {}
        strongest_selected_score = scores[selected[-1]]
        alpha = {
            symbol: max(strongest_selected_score - scores[symbol], 0.0)
            for symbol in selected
        }
        if sum(alpha.values()) == 0:
            alpha = {symbol: 1.0 for symbol in selected}
        gross = sum(abs(value) for value in alpha.values())
        target_gross = min(context.max_gross_exposure, 1.0)
        return {symbol: value * target_gross / gross for symbol, value in alpha.items() if value}
"""

NEUTRALIZATION_TEMPLATE = """def group_for(symbol, stock):
    # stock is the metadata row from Available Stocks for this symbol.
    # Common fields: symbol, name, region, sector, industry, exchange.
    region = stock.get("region") or "Other"
    sector = stock.get("sector") or "Other"
    industry = stock.get("industry") or "Other"

    # Custom grouping examples. Return any label you want neutralized together.
    # if region in {{"China", "Japan", "Taiwan"}}:
    #     return "Asia"
    # if sector in {{"Technology", "Communication"}}:
    #     return "Growth"

    {group_logic}


def neutralize(alpha, groups):
    # alpha maps symbol -> raw signal weight.
    # groups maps symbol -> the bucket returned by group_for.
    # A bucket with one stock becomes 0 because there is no within-group relative bet.
    adjusted = {{}}
    for group in set(groups.values()):
        symbols = [symbol for symbol, item_group in groups.items() if item_group == group and symbol in alpha]
        if not symbols:
            continue
        mean_alpha = sum(alpha.get(symbol, 0.0) for symbol in symbols) / len(symbols)
        for symbol in symbols:
            adjusted[symbol] = alpha.get(symbol, 0.0) - mean_alpha
    return adjusted
"""

NEUTRALIZATION_BY_MARKET_CODE = NEUTRALIZATION_TEMPLATE.format(
    group_logic='# Market neutralization puts every stock in one bucket.\n    return "Market"'
)
NEUTRALIZATION_BY_REGION_CODE = NEUTRALIZATION_TEMPLATE.format(
    group_logic="# Region neutralization groups by issuer country/region.\n    return region"
)
NEUTRALIZATION_BY_SECTOR_CODE = NEUTRALIZATION_TEMPLATE.format(
    group_logic="# Sector neutralization groups by broad sector.\n    return sector"
)
NEUTRALIZATION_BY_INDUSTRY_CODE = NEUTRALIZATION_TEMPLATE.format(
    group_logic="# Industry neutralization groups by the more specific industry field.\n    return industry"
)
DEFAULT_NEUTRALIZATION_CODE = NEUTRALIZATION_BY_MARKET_CODE

DEFAULT_PREPROCESS_CODE = """def preprocess(data, symbols, context):
    # No preprocessing: use the market data exactly as loaded.
    return data
"""

DEFAULT_BACKFILL_CODE = """def preprocess(data, symbols, context):
    # DataFrame columns: date, symbol, open, high, low, close, volume.
    # Backfill uses the next available future value within each symbol.
    if pd is None:
        raise ImportError("Install pandas/numpy for DataFrame preprocessing: pip install pandas numpy")

    frame = data.to_frame()
    if frame.empty:
        return data

    selected = list(symbols or data.symbols)
    dates = sorted(frame["date"].unique())
    full_index = pd.MultiIndex.from_product([dates, selected], names=["date", "symbol"])
    panel = frame.set_index(["date", "symbol"]).sort_index().reindex(full_index)

    price_cols = ["open", "high", "low", "close"]
    panel[price_cols] = panel.groupby(level="symbol")[price_cols].bfill()
    panel["volume"] = panel["volume"].fillna(0.0)

    filled = panel.dropna(subset=["close"]).reset_index()
    for column in ["open", "high", "low"]:
        filled[column] = filled[column].fillna(filled["close"])
    return MarketData.from_frame(filled)
"""

DEFAULT_FORWARD_FILL_CODE = """def preprocess(data, symbols, context):
    # DataFrame columns: date, symbol, open, high, low, close, volume.
    # Forward-fill uses the most recent past value within each symbol.
    if pd is None:
        raise ImportError("Install pandas/numpy for DataFrame preprocessing: pip install pandas numpy")

    frame = data.to_frame()
    if frame.empty:
        return data

    selected = list(symbols or data.symbols)
    dates = sorted(frame["date"].unique())
    full_index = pd.MultiIndex.from_product([dates, selected], names=["date", "symbol"])
    panel = frame.set_index(["date", "symbol"]).sort_index().reindex(full_index)

    price_cols = ["open", "high", "low", "close"]
    panel[price_cols] = panel.groupby(level="symbol")[price_cols].ffill()
    panel["volume"] = panel["volume"].fillna(0.0)

    filled = panel.dropna(subset=["close"]).reset_index()
    for column in ["open", "high", "low"]:
        filled[column] = filled[column].fillna(filled["close"])
    return MarketData.from_frame(filled)
"""

DEFAULT_STRATEGIES = [
    {"id": "none", "name": "None: Use Portfolio Weights", "code": DEFAULT_CODE},
    {"id": "momentum", "name": "Momentum", "code": DEFAULT_MOMENTUM_CODE},
    {"id": "reversion", "name": "Mean Reversion", "code": DEFAULT_REVERSION_CODE},
]

DEFAULT_PREPROCESSORS = [
    {"id": "none", "name": "None: Use Raw Data", "code": DEFAULT_PREPROCESS_CODE},
    {"id": "backfill", "name": "Backfill Missing Bars", "code": DEFAULT_BACKFILL_CODE},
    {"id": "forward_fill", "name": "Forward Fill Missing Bars", "code": DEFAULT_FORWARD_FILL_CODE},
]

DEFAULT_NEUTRALIZERS = [
    {"id": "none", "name": "None", "code": "def neutralize(alpha, groups):\n    return dict(alpha)\n"},
    {"id": "market", "name": "Market", "code": NEUTRALIZATION_BY_MARKET_CODE},
    {"id": "region", "name": "Region", "code": NEUTRALIZATION_BY_REGION_CODE},
    {"id": "sector", "name": "Sector", "code": NEUTRALIZATION_BY_SECTOR_CODE},
    {"id": "industry", "name": "Industry", "code": NEUTRALIZATION_BY_INDUSTRY_CODE},
]

DATA_CACHE_PATH = Path(__file__).resolve().parents[2] / "data" / "yahoo_prices.csv"
CATALOG_CACHE_PATH = Path(__file__).resolve().parents[2] / "data" / "catalogs" / "yahoo_us_catalog.json"
STRATEGY_LIBRARY_PATH = Path(__file__).resolve().parents[2] / "strategies" / "strategy_library.json"
PREPROCESSING_LIBRARY_PATH = Path(__file__).resolve().parents[2] / "strategies" / "preprocessing_library.json"
NEUTRALIZATION_LIBRARY_PATH = Path(__file__).resolve().parents[2] / "strategies" / "neutralization_library.json"


def main() -> None:
    parser = argparse.ArgumentParser(prog="portfoliolab-app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), PortfolioLabHandler)
    print(f"PortfolioLab app: http://{args.host}:{args.port}")
    server.serve_forever()


class PortfolioLabHandler(BaseHTTPRequestHandler):
    data: MarketData | None = None
    data_source = "Yahoo Finance"
    data_message = "Not loaded"
    catalog: list[StockInfo] | None = None

    def do_GET(self) -> None:
        if self.path == "/" or self.path.startswith("/?"):
            self._send_html(INDEX_HTML)
            return
        if self.path == "/api/pools":
            try:
                if self.__class__.data is None and DATA_CACHE_PATH.exists():
                    result = _load_initial_market_data()
                    self.__class__.data = result["data"]
                    self.__class__.data_source = result["source"]
                    self.__class__.data_message = result["message"]
                self._send_json(_catalog_payload(self.__class__.data, self._ensure_catalog(), self.data_source, self.data_message))
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        if self.path == "/api/strategies":
            try:
                self._send_json(_strategy_library_payload())
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        if self.path == "/api/preprocessors":
            try:
                self._send_json(_preprocessing_library_payload())
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        if self.path == "/api/neutralizers":
            try:
                self._send_json(_neutralization_library_payload())
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        if self.path == "/api/refresh-data":
            try:
                payload = self._read_json()
                result = _load_price_data_payload(payload, self._ensure_catalog(), self.__class__.data)
                self.__class__.data = result["data"]
                self.__class__.data_source = result["source"]
                self.__class__.data_message = result["message"]
                self._send_json(
                    {
                        "catalog": _catalog_payload(result["data"], self._ensure_catalog(), self.data_source, self.data_message),
                        "message": result["message"],
                    }
                )
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        if self.path == "/api/refresh-catalog":
            try:
                payload = self._read_json()
                result = _refresh_catalog_payload(payload)
                self.__class__.catalog = result["stocks"]
                self.__class__.data_source = result["source"]
                self.__class__.data_message = result["message"]
                self._send_json(
                    {
                        "catalog": _catalog_payload(self.__class__.data, result["stocks"], self.data_source, self.data_message),
                        "message": result["message"],
                    }
                )
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        if self.path == "/api/lookup-symbols":
            try:
                result = _lookup_symbols_payload(self._read_json())
                self.__class__.catalog = _merge_stock_catalog(self._ensure_catalog(), _stocks_from_payload(result["stocks"]))
                self._send_json(result)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        if self.path == "/api/load-price-data":
            try:
                payload = self._read_json()
                result = _load_price_data_payload(payload, self._ensure_catalog(), self.__class__.data)
                self.__class__.data = result["data"]
                self.__class__.data_source = result["source"]
                self.__class__.data_message = result["message"]
                self._send_json(
                    {
                        "catalog": _catalog_payload(result["data"], self._ensure_catalog(), self.data_source, self.data_message),
                        "message": result["message"],
                    }
                )
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        if self.path == "/api/strategies/save":
            try:
                result = _save_strategy_payload(self._read_json())
                self._send_json(result)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        if self.path == "/api/strategies/delete":
            try:
                result = _delete_strategy_payload(self._read_json())
                self._send_json(result)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        if self.path == "/api/preprocessors/save":
            try:
                result = _save_preprocessor_payload(self._read_json())
                self._send_json(result)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        if self.path == "/api/preprocessors/delete":
            try:
                result = _delete_preprocessor_payload(self._read_json())
                self._send_json(result)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        if self.path == "/api/neutralizers/save":
            try:
                result = _save_neutralizer_payload(self._read_json())
                self._send_json(result)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        if self.path == "/api/neutralizers/delete":
            try:
                result = _delete_neutralizer_payload(self._read_json())
                self._send_json(result)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        if self.path != "/api/backtest":
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        try:
            payload = self._read_json()
            result = _run_payload_backtest(self._ensure_data(), payload)
            self._send_json(result)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    def log_message(self, format: str, *args: Any) -> None:
        return

    @classmethod
    def _ensure_data(cls) -> MarketData:
        if cls.data is None:
            result = _load_initial_market_data()
            cls.data = result["data"]
            cls.data_source = result["source"]
            cls.data_message = result["message"]
        return cls.data

    @classmethod
    def _ensure_catalog(cls) -> list[StockInfo]:
        if cls.catalog is None:
            cls.catalog = _load_initial_catalog()
        return cls.catalog

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("content-length", "0"))
        raw = self.rfile.read(length)
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def _send_html(self, content: str) -> None:
        body = content.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _catalog_payload(
    data: MarketData | None = None,
    stocks: list[StockInfo] | None = None,
    source: str = "Yahoo Finance",
    message: str = "",
) -> dict[str, Any]:
    stocks = stocks or STOCKS
    dates = data.window_dates() if data is not None else []
    default_start = parse_date("2019-01-02")
    default_end = dates[-1] if dates else date.today()
    default_split = _default_split_date(data, default_start, default_end) if data is not None else default_start
    default_config = BacktestConfig(start=default_start, end=default_end)
    loaded_symbols = set(data.symbols) if data is not None else set()
    stock_by_symbol = {stock.symbol: stock for stock in stocks}
    for symbol in sorted(loaded_symbols.difference(stock_by_symbol)):
        stock_by_symbol[symbol] = StockInfo(symbol, symbol, "Unknown", source, "")
    stocks = sorted(stock_by_symbol.values(), key=lambda stock: stock.symbol)
    return {
        "stocks": [_stock_payload(stock, data, default_start, default_end) for stock in stocks],
        "rawPrices": _price_payload(data, sorted(loaded_symbols), default_config) if data is not None else {},
        "rawBars": _bar_payload(data, sorted(loaded_symbols), default_config) if data is not None else {},
        "pools": _build_pools(stocks),
        "data": {
            "source": source,
            "message": message,
            "start": dates[0].isoformat() if dates else "",
            "end": dates[-1].isoformat() if dates else "",
            "symbols": sorted(loaded_symbols),
            "loadedCount": len(loaded_symbols),
            "catalogCount": len(stocks),
        },
        "defaults": {
            "start": default_start.isoformat(),
            "end": default_end.isoformat(),
            "splitDate": default_split.isoformat(),
            "rebalanceFrequency": "monthly",
            "selectedPool": _default_pool_id(stocks),
            "code": DEFAULT_CODE,
        },
    }


def _refresh_data_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return _load_price_data_payload(payload, STOCKS)


def _refresh_catalog_payload(payload: dict[str, Any]) -> dict[str, Any]:
    source = payload.get("source", "yahoo")
    if source != "yahoo":
        raise DataProviderError("Supported catalog source: yahoo")

    try:
        rows = fetch_nasdaq_us_symbol_catalog()
        stocks = [_stock_from_catalog_row(row) for row in rows]
        _write_catalog_cache(stocks)
        message = f"Refreshed {len(stocks)} U.S. symbols from Nasdaq Trader."
    except DataProviderError as exc:
        stocks = _load_catalog_cache() or STOCKS
        message = f"{exc}. Using cached/seed catalog with {len(stocks)} symbols."
    return {
        "stocks": stocks,
        "source": "Yahoo Finance",
        "message": message,
    }


def _lookup_symbols_payload(payload: dict[str, Any]) -> dict[str, Any]:
    source = payload.get("source", "yahoo")
    if source != "yahoo":
        raise DataProviderError("Supported symbol source: yahoo")

    symbols = _parse_symbol_input(payload.get("symbols", ""))
    rows = lookup_yahoo_symbols(symbols)
    stocks = [_stock_from_catalog_row(row) for row in rows]
    return {
        "stocks": [_stock_info_payload(stock) for stock in stocks],
        "source": "Yahoo Finance",
        "message": f"Found {len(stocks)} available symbol{'s' if len(stocks) != 1 else ''}.",
    }


def _load_price_data_payload(
    payload: dict[str, Any],
    catalog: list[StockInfo],
    existing_data: MarketData | None = None,
) -> dict[str, Any]:
    source = payload.get("source", "yahoo")
    if source != "yahoo":
        raise DataProviderError("Supported data source: yahoo")

    symbols = [str(symbol).strip().upper() for symbol in payload.get("symbols", []) if str(symbol).strip()]
    if not symbols:
        raise DataProviderError("Add at least one available stock before refreshing price data")
    catalog = _merge_stock_catalog(catalog, _stocks_from_payload(payload.get("stocks", [])))
    catalog_by_symbol = {stock.symbol: stock for stock in catalog}
    missing = [symbol for symbol in symbols if symbol not in catalog_by_symbol]
    if missing:
        raise DataProviderError(f"Symbols are not in the active catalog: {', '.join(missing[:8])}")
    period = payload.get("period", "10y")
    symbol_map = {
        symbol: catalog_by_symbol[symbol].provider_symbol or symbol
        for symbol in symbols
        if symbol in catalog_by_symbol
    }
    data = fetch_yahoo_data(symbol_map, period=period)
    write_market_data_csv(data, DATA_CACHE_PATH)
    return {
        "data": data,
        "source": "Yahoo Finance",
        "message": f"Refreshed {len(data.symbols)} available symbols from Yahoo Finance through {data.dates[-1].isoformat()}",
    }


def _parse_symbol_input(value: Any) -> list[str]:
    if isinstance(value, list):
        raw_values = [str(item) for item in value]
    else:
        raw_values = re.split(r"[\s,;]+", str(value or ""))
    return [symbol.strip().upper() for symbol in raw_values if symbol.strip()]


def _stocks_from_payload(rows: Any) -> list[StockInfo]:
    if not isinstance(rows, list):
        return []
    stocks = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("symbol", "")).strip().upper()
        if not symbol:
            continue
        stocks.append(_stock_from_catalog_row({**row, "symbol": symbol}))
    return stocks


def _merge_stock_catalog(base: list[StockInfo], additions: list[StockInfo]) -> list[StockInfo]:
    merged = {stock.symbol: stock for stock in base}
    for stock in additions:
        merged[stock.symbol] = stock
    return sorted(merged.values(), key=lambda stock: stock.symbol)


def _strategy_library_payload() -> dict[str, Any]:
    return {
        "strategies": _load_strategy_library(),
        "path": str(STRATEGY_LIBRARY_PATH),
    }


def _save_strategy_payload(payload: dict[str, Any]) -> dict[str, Any]:
    strategies = _load_strategy_library()
    name = str(payload.get("name", "")).strip()
    code = str(payload.get("code", "")).strip()
    current_id = str(payload.get("id", "")).strip()
    save_as_new = bool(payload.get("saveAsNew"))
    if not name:
        raise ValueError("Strategy name is required")
    if not code:
        raise ValueError("Strategy code is required")
    duplicate = next(
        (
            strategy
            for strategy in strategies
            if strategy["name"].casefold() == name.casefold()
            and (save_as_new or strategy["id"] != current_id)
        ),
        None,
    )
    if duplicate:
        raise ValueError(f"Strategy name already exists: {name}")

    if save_as_new or not current_id:
        saved = {"id": _unique_strategy_id(name, strategies), "name": name, "code": code}
        strategies.append(saved)
    else:
        saved = None
        for strategy in strategies:
            if strategy["id"] == current_id:
                strategy["name"] = name
                strategy["code"] = code
                saved = strategy
                break
        if saved is None:
            saved = {"id": _unique_strategy_id(name, strategies), "name": name, "code": code}
            strategies.append(saved)

    _write_strategy_library(strategies)
    return {
        "strategies": strategies,
        "selectedId": saved["id"],
        "message": "Saved as a new strategy." if save_as_new else "Strategy updated.",
        "path": str(STRATEGY_LIBRARY_PATH),
    }


def _delete_strategy_payload(payload: dict[str, Any]) -> dict[str, Any]:
    strategies = _load_strategy_library()
    strategy_id = str(payload.get("id", "")).strip()
    if len(strategies) <= 1:
        raise ValueError("Keep at least one strategy")
    next_strategies = [strategy for strategy in strategies if strategy["id"] != strategy_id]
    if len(next_strategies) == len(strategies):
        raise ValueError("Strategy was not found")
    _write_strategy_library(next_strategies)
    return {
        "strategies": next_strategies,
        "selectedId": next_strategies[0]["id"],
        "message": "Strategy deleted.",
        "path": str(STRATEGY_LIBRARY_PATH),
    }


def _preprocessing_library_payload() -> dict[str, Any]:
    return {
        "preprocessors": _load_preprocessing_library(),
        "path": str(PREPROCESSING_LIBRARY_PATH),
    }


def _save_preprocessor_payload(payload: dict[str, Any]) -> dict[str, Any]:
    preprocessors = _load_preprocessing_library()
    load_inline_preprocessor(str(payload.get("code", "")).strip())
    saved, preprocessors, message = _save_code_library_item(payload, preprocessors, "Preprocessing")
    _write_code_library(PREPROCESSING_LIBRARY_PATH, "preprocessors", preprocessors)
    return {
        "preprocessors": preprocessors,
        "selectedId": saved["id"],
        "message": message,
        "path": str(PREPROCESSING_LIBRARY_PATH),
    }


def _delete_preprocessor_payload(payload: dict[str, Any]) -> dict[str, Any]:
    preprocessors = _load_preprocessing_library()
    preprocessor_id = str(payload.get("id", "")).strip()
    if len(preprocessors) <= 1:
        raise ValueError("Keep at least one preprocessor")
    next_preprocessors = [preprocessor for preprocessor in preprocessors if preprocessor["id"] != preprocessor_id]
    if len(next_preprocessors) == len(preprocessors):
        raise ValueError("Preprocessor was not found")
    _write_code_library(PREPROCESSING_LIBRARY_PATH, "preprocessors", next_preprocessors)
    return {
        "preprocessors": next_preprocessors,
        "selectedId": next_preprocessors[0]["id"],
        "message": "Preprocessor deleted.",
        "path": str(PREPROCESSING_LIBRARY_PATH),
    }


def _neutralization_library_payload() -> dict[str, Any]:
    return {
        "neutralizers": _load_neutralization_library(),
        "path": str(NEUTRALIZATION_LIBRARY_PATH),
    }


def _save_neutralizer_payload(payload: dict[str, Any]) -> dict[str, Any]:
    neutralizers = _load_neutralization_library()
    load_inline_neutralizer(str(payload.get("code", "")).strip())
    saved, neutralizers, message = _save_code_library_item(payload, neutralizers, "Neutralization")
    _write_code_library(NEUTRALIZATION_LIBRARY_PATH, "neutralizers", neutralizers)
    return {
        "neutralizers": neutralizers,
        "selectedId": saved["id"],
        "message": message,
        "path": str(NEUTRALIZATION_LIBRARY_PATH),
    }


def _delete_neutralizer_payload(payload: dict[str, Any]) -> dict[str, Any]:
    neutralizers = _load_neutralization_library()
    neutralizer_id = str(payload.get("id", "")).strip()
    if len(neutralizers) <= 1:
        raise ValueError("Keep at least one neutralizer")
    next_neutralizers = [neutralizer for neutralizer in neutralizers if neutralizer["id"] != neutralizer_id]
    if len(next_neutralizers) == len(neutralizers):
        raise ValueError("Neutralizer was not found")
    _write_code_library(NEUTRALIZATION_LIBRARY_PATH, "neutralizers", next_neutralizers)
    return {
        "neutralizers": next_neutralizers,
        "selectedId": next_neutralizers[0]["id"],
        "message": "Neutralizer deleted.",
        "path": str(NEUTRALIZATION_LIBRARY_PATH),
    }


def _load_neutralization_library() -> list[dict[str, str]]:
    loaded = _load_code_library(NEUTRALIZATION_LIBRARY_PATH, DEFAULT_NEUTRALIZERS)
    default_ids = {item["id"] for item in DEFAULT_NEUTRALIZERS}
    neutralizers = [dict(item) for item in DEFAULT_NEUTRALIZERS]
    neutralizers.extend(item for item in loaded if item["id"] not in default_ids and item["id"] != "subindustry")
    return neutralizers


def _load_preprocessing_library() -> list[dict[str, str]]:
    return _load_code_library(PREPROCESSING_LIBRARY_PATH, DEFAULT_PREPROCESSORS)


def _load_strategy_library() -> list[dict[str, str]]:
    if not STRATEGY_LIBRARY_PATH.exists():
        return [dict(strategy) for strategy in DEFAULT_STRATEGIES]
    raw = json.loads(STRATEGY_LIBRARY_PATH.read_text(encoding="utf-8"))
    entries = raw.get("strategies", raw) if isinstance(raw, dict) else raw
    if not isinstance(entries, list):
        return [dict(strategy) for strategy in DEFAULT_STRATEGIES]
    strategies = []
    seen_ids = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name", "")).strip()
        code = str(entry.get("code", "")).strip()
        if not name or not code:
            continue
        strategy_id = str(entry.get("id", "")).strip() or _unique_strategy_id(name, strategies)
        if strategy_id in seen_ids:
            strategy_id = _unique_strategy_id(name, strategies)
        seen_ids.add(strategy_id)
        strategies.append({"id": strategy_id, "name": name, "code": code})
    return strategies or [dict(strategy) for strategy in DEFAULT_STRATEGIES]


def _write_strategy_library(strategies: list[dict[str, str]]) -> None:
    STRATEGY_LIBRARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    STRATEGY_LIBRARY_PATH.write_text(
        json.dumps({"strategies": strategies}, indent=2) + "\n",
        encoding="utf-8",
    )


def _load_code_library(path: Path, defaults: list[dict[str, str]]) -> list[dict[str, str]]:
    if not path.exists():
        return [dict(item) for item in defaults]
    raw = json.loads(path.read_text(encoding="utf-8"))
    entries = raw.get("items") or raw.get("neutralizers") or raw.get("strategies") or raw if isinstance(raw, dict) else raw
    if not isinstance(entries, list):
        return [dict(item) for item in defaults]
    items = []
    seen_ids = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name", "")).strip()
        code = str(entry.get("code", "")).strip()
        if not name or not code:
            continue
        item_id = str(entry.get("id", "")).strip() or _unique_strategy_id(name, items)
        if item_id in seen_ids:
            item_id = _unique_strategy_id(name, items)
        seen_ids.add(item_id)
        items.append({"id": item_id, "name": name, "code": code})
    return items or [dict(item) for item in defaults]


def _save_code_library_item(
    payload: dict[str, Any],
    items: list[dict[str, str]],
    label: str,
) -> tuple[dict[str, str], list[dict[str, str]], str]:
    name = str(payload.get("name", "")).strip()
    code = str(payload.get("code", "")).strip()
    current_id = str(payload.get("id", "")).strip()
    save_as_new = bool(payload.get("saveAsNew"))
    if not name:
        raise ValueError(f"{label} name is required")
    if not code:
        raise ValueError(f"{label} code is required")
    duplicate = next(
        (
            item
            for item in items
            if item["name"].casefold() == name.casefold()
            and (save_as_new or item["id"] != current_id)
        ),
        None,
    )
    if duplicate:
        raise ValueError(f"{label} name already exists: {name}")
    if save_as_new or not current_id:
        saved = {"id": _unique_strategy_id(name, items), "name": name, "code": code}
        items.append(saved)
    else:
        saved = None
        for item in items:
            if item["id"] == current_id:
                item["name"] = name
                item["code"] = code
                saved = item
                break
        if saved is None:
            saved = {"id": _unique_strategy_id(name, items), "name": name, "code": code}
            items.append(saved)
    return saved, items, f"{label} saved as new." if save_as_new else f"{label} updated."


def _write_code_library(path: Path, key: str, items: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({key: items}, indent=2) + "\n", encoding="utf-8")


def _strategy_slug(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug or "strategy"


def _unique_strategy_id(name: str, strategies: list[dict[str, str]]) -> str:
    base = _strategy_slug(name)
    existing = {strategy["id"] for strategy in strategies}
    strategy_id = base
    counter = 2
    while strategy_id in existing:
        strategy_id = f"{base}_{counter}"
        counter += 1
    return strategy_id


def _load_initial_market_data() -> dict[str, Any]:
    if DATA_CACHE_PATH.exists():
        data = MarketData.from_csv(DATA_CACHE_PATH)
        return {
            "data": data,
            "source": "Yahoo Finance Cache",
            "message": f"Loaded cached Yahoo Finance data through {data.dates[-1].isoformat()}",
        }
    raise DataProviderError("Load price data for the selected basket before running a backtest.")


def _load_initial_catalog() -> list[StockInfo]:
    return _merge_stock_catalog(_load_catalog_cache(), STOCKS)


def _stock_from_catalog_row(row: dict[str, str]) -> StockInfo:
    return StockInfo(
        symbol=str(row.get("symbol", "")).strip().upper(),
        name=str(row.get("name", "")).strip() or str(row.get("symbol", "")).strip().upper(),
        region=str(row.get("region", "")).strip() or "United States",
        source=str(row.get("source", "")).strip() or "Yahoo Finance",
        sector=str(row.get("sector", "")).strip(),
        exchange=str(row.get("exchange", "")).strip(),
        provider_symbol=str(row.get("provider_symbol", "")).strip() or None,
        industry=str(row.get("industry", "")).strip(),
    )


def _load_catalog_cache() -> list[StockInfo]:
    if not CATALOG_CACHE_PATH.exists():
        return []
    raw = json.loads(CATALOG_CACHE_PATH.read_text(encoding="utf-8"))
    entries = raw.get("stocks", raw) if isinstance(raw, dict) else raw
    if not isinstance(entries, list):
        return []
    stocks = [_stock_from_catalog_row(entry) for entry in entries if isinstance(entry, dict)]
    return [stock for stock in stocks if _is_common_equity_stock(stock)]


def _write_catalog_cache(stocks: list[StockInfo]) -> None:
    CATALOG_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CATALOG_CACHE_PATH.write_text(
        json.dumps({"stocks": [_stock_info_payload(stock) for stock in stocks]}, indent=2) + "\n",
        encoding="utf-8",
    )


def _stock_info_payload(stock: StockInfo) -> dict[str, str]:
    return {
        "symbol": stock.symbol,
        "name": stock.name,
        "region": stock.region,
        "source": stock.source,
        "sector": stock.sector,
        "industry": stock.industry or STOCK_INDUSTRIES.get(stock.symbol, ""),
        "exchange": stock.exchange,
        "provider_symbol": stock.provider_symbol or stock.symbol,
    }


def _is_common_equity_stock(stock: StockInfo) -> bool:
    if not stock.symbol or "$" in stock.symbol or "^" in stock.symbol:
        return False
    lower_name = stock.name.casefold()
    excluded_terms = (
        "preferred",
        "preference",
        "warrant",
        "right",
        "unit",
        "note",
        "notes",
        "bond",
        "debenture",
        "perpetual",
        "redeemable",
        "callable",
        "trust preferred",
        "closed-end",
        "closed end",
        "exchange traded fund",
        " etf",
        " etn",
    )
    return not any(term in lower_name for term in excluded_terms)


def _build_pools(stocks: list[StockInfo]) -> list[dict[str, Any]]:
    available = {stock.symbol for stock in stocks}
    pools = [
        {"id": "all", "name": "All Catalog", "symbols": sorted(available)},
    ]
    existing_names = {"all catalog"}
    for pool in POOLS:
        symbols = [symbol for symbol in pool["symbols"] if symbol in available]
        if symbols:
            pools.append({"id": pool["id"], "name": pool["name"], "symbols": symbols})
            existing_names.add(pool["name"].casefold())
    for region in sorted({stock.region for stock in stocks if stock.region}):
        if region.casefold() in existing_names:
            continue
        symbols = sorted(stock.symbol for stock in stocks if stock.region == region)
        if symbols:
            pools.append({"id": f"region_{_strategy_slug(region)}", "name": region, "symbols": symbols})
    return pools


def _default_pool_id(stocks: list[StockInfo]) -> str:
    available = {stock.symbol for stock in stocks}
    if any(pool["id"] == "global_quality" and set(pool["symbols"]).issubset(available) for pool in POOLS):
        return "global_quality"
    return "all"


def _default_split_date(data: MarketData, start: date, end: date) -> date:
    dates = data.window_dates(start, end)
    if len(dates) < 5:
        return start
    return dates[max(0, min(len(dates) - 2, math.floor((len(dates) - 1) * 0.8)))]


def _stock_payload(stock: StockInfo, data: MarketData | None, start: date, end: date) -> dict[str, Any]:
    dates = data.window_dates(start, end) if data is not None and stock.symbol in data.symbols else []
    first = next((data.close(day, stock.symbol) for day in dates if data and data.close(day, stock.symbol)), None)
    last = next((data.close(day, stock.symbol) for day in reversed(dates) if data and data.close(day, stock.symbol)), None)
    total_return = (last / first) - 1.0 if first and last else 0.0
    return {
        "symbol": stock.symbol,
        "name": stock.name,
        "region": stock.region,
        "source": stock.source,
        "sector": stock.sector,
        "industry": stock.industry or STOCK_INDUSTRIES.get(stock.symbol, ""),
        "exchange": stock.exchange,
        "providerSymbol": stock.provider_symbol or stock.symbol,
        "loaded": bool(dates),
        "firstClose": first,
        "lastClose": last,
        "totalReturn": total_return,
    }


def _run_payload_backtest(data: MarketData, payload: dict[str, Any]) -> dict[str, Any]:
    selected = [symbol.upper() for symbol in payload.get("symbols", []) if symbol]
    if not selected:
        raise ValueError("Select at least one stock")
    stock_catalog = _merge_stock_catalog(STOCKS, _stocks_from_payload(payload.get("stocks", [])))
    preprocessor = load_inline_preprocessor(payload.get("preprocessingCode", DEFAULT_PREPROCESS_CODE))
    data = preprocessor(
        data,
        selected,
        {
            "symbols": list(selected),
            "stocks": [_stock_info_payload(stock) for stock in stock_catalog if stock.symbol in selected],
        },
    )
    if not isinstance(data, MarketData):
        raise ValueError("preprocess(data, symbols, context) must return MarketData")

    legacy_mode = payload.get("mode")
    portfolio_mode = payload.get("portfolioMode", legacy_mode or "equal")
    signal_strategy = payload.get("signalStrategy", "none")
    if legacy_mode in {"momentum", "custom"} and payload.get("signalStrategy") is None:
        signal_strategy = legacy_mode
    raw_weights = {symbol.upper(): float(weight) for symbol, weight in payload.get("weights", {}).items()}
    base_weights = normalize_weights({symbol: raw_weights.get(symbol, 0.0) for symbol in selected})
    if portfolio_mode == "equal" or not base_weights:
        base_weights = equal_weights(selected)

    if signal_strategy == "none":
        strategy = FixedWeightStrategy(base_weights)
    elif signal_strategy == "momentum":
        strategy = SelectedMomentumStrategy(selected, top_n=min(5, len(selected)))
    elif signal_strategy == "reversion":
        strategy = SelectedMeanReversionStrategy(selected, bottom_n=min(5, len(selected)))
    elif signal_strategy == "custom":
        strategy = load_inline_strategy(payload.get("code", DEFAULT_CODE), selected, base_weights)
    elif portfolio_mode == "manual":
        strategy = FixedWeightStrategy(base_weights)
    else:
        base_weights = equal_weights(selected)
        strategy = FixedWeightStrategy(equal_weights(selected))

    neutralization_mode = payload.get("neutralizationMode", "none")
    if neutralization_mode != "none":
        neutralize, group_for = load_inline_neutralizer(payload.get("neutralizationCode", DEFAULT_NEUTRALIZATION_CODE))
        strategy = NeutralizedStrategy(
            strategy=strategy,
            neutralize=neutralize,
            groups=_neutralization_groups(selected, neutralization_mode, stock_catalog, group_for),
        )

    config = BacktestConfig(
        start=parse_date(payload.get("start", "2019-01-02")),
        end=parse_date(payload.get("end", data.dates[-1].isoformat())),
        rebalance_frequency=payload.get("rebalanceFrequency", "monthly"),
        transaction_cost_bps=float(payload.get("transactionCostBps", 5.0)),
        slippage_bps=float(payload.get("slippageBps", 2.0)),
    )
    split_date = parse_date(payload.get("splitDate", _default_split_date(data, config.start, config.end).isoformat()))
    evaluation = run_is_os_evaluation(data, strategy, config, split_date)
    full_metrics = evaluate(evaluation.full)

    dates = [point.date for point in evaluation.full.equity_curve]
    return {
        "metrics": {
            "full": _metrics_payload(full_metrics),
            "is": _metrics_payload(evaluation.in_sample_metrics),
            "os": _metrics_payload(evaluation.out_of_sample_metrics),
            "training": _metrics_payload(evaluation.in_sample_metrics),
            "testing": _metrics_payload(evaluation.out_of_sample_metrics),
        },
        "yearly": _yearly_payload(evaluation.full),
        "periods": {
            "start": dates[0].isoformat(),
            "end": dates[-1].isoformat(),
            "splitDate": split_date.isoformat(),
        },
        "equity": [
            {
                "date": point.date.isoformat(),
                "value": point.value,
                "dailyPnl": point.daily_pnl,
                "dailyReturn": point.daily_return,
                "turnover": point.turnover,
                "dollarsTraded": point.dollars_traded,
                "grossExposure": point.gross_exposure,
            }
            for point in evaluation.full.equity_curve
        ],
        "capital": {
            "halfBookSize": config.initial_cash,
            "bookSize": config.initial_cash * 2.0,
        },
        "prices": _price_payload(data, selected, config),
        "weights": _weights_payload(evaluation.full, selected),
        "stocks": [_stock_summary(data, symbol, config, evaluation.full, stock_catalog) for symbol in selected],
        "finalWeights": evaluation.full.final_weights,
    }


def _metrics_payload(metrics) -> dict[str, float]:
    return metrics.as_dict()


def _neutralization_groups(selected: list[str], mode: str, stock_catalog: list[StockInfo] | None = None, group_for=None) -> dict[str, str]:
    stock_by_symbol = {stock.symbol: stock for stock in stock_catalog or []}
    if group_for is not None:
        groups = {}
        for symbol in selected:
            stock = _stock_info_payload(stock_by_symbol[symbol]) if symbol in stock_by_symbol else {"symbol": symbol}
            groups[symbol] = str(group_for(symbol, stock) or "Other")
        return groups
    if mode == "market":
        return {symbol: "Market" for symbol in selected}
    if mode == "region":
        return {
            symbol: stock_by_symbol.get(symbol).region if stock_by_symbol.get(symbol) and stock_by_symbol[symbol].region else "Other"
            for symbol in selected
        }
    if mode == "sector":
        return {
            symbol: stock_by_symbol.get(symbol).sector if stock_by_symbol.get(symbol) and stock_by_symbol[symbol].sector else STOCK_SECTORS.get(symbol, "Other")
            for symbol in selected
        }
    if mode == "industry":
        return {
            symbol: stock_by_symbol.get(symbol).industry if stock_by_symbol.get(symbol) and stock_by_symbol[symbol].industry else STOCK_INDUSTRIES.get(symbol, "Other")
            for symbol in selected
        }
    return {symbol: "All" for symbol in selected}


def _yearly_payload(result) -> list[dict[str, Any]]:
    rows = []
    years = sorted({point.date.year for point in result.equity_curve})
    for year in years:
        points = [point for point in result.equity_curve if point.date.year == year]
        row = _period_performance_row(year, points, result.config.initial_cash)
        if row:
            rows.append(row)
    return rows


def _period_performance_row(label: int | str, points, half_book_size: float) -> dict[str, Any] | None:
    if not points:
        return None
    returns = [point.daily_return for point in points]
    daily_pnls = [point.daily_pnl for point in points]
    annualized_return = _ratio((sum(daily_pnls) / len(daily_pnls)) * 252.0, half_book_size)
    daily_volatility = _stddev(returns)
    sharpe = _ratio(sum(returns) / len(returns), daily_volatility) * math.sqrt(252.0)
    turnover = sum(point.turnover for point in points) / len(points)
    drawdown = _pnl_drawdown(daily_pnls, half_book_size)
    margin = _ratio(sum(daily_pnls), sum(point.dollars_traded for point in points))
    long_count = round(sum(sum(1 for weight in point.weights.values() if weight > 0) for point in points) / len(points))
    short_count = round(sum(sum(1 for weight in point.weights.values() if weight < 0) for point in points) / len(points))
    fitness = sharpe * math.sqrt(abs(annualized_return) / max(turnover, 0.125))
    return {
        "year": label,
        "sharpe": sharpe,
        "turnover": turnover,
        "fitness": fitness,
        "returns": annualized_return,
        "drawdown": drawdown,
        "margin": margin,
        "longCount": long_count,
        "shortCount": short_count,
    }


def _stddev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((item - mean) ** 2 for item in values) / (len(values) - 1)
    return math.sqrt(variance)


def _ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _max_drawdown(values: list[float]) -> float:
    peak = values[0]
    drawdown = 0.0
    for value in values:
        peak = max(peak, value)
        if peak:
            drawdown = min(drawdown, (value / peak) - 1.0)
    return drawdown


def _pnl_drawdown(daily_pnls: list[float], half_book_size: float) -> float:
    cumulative = 0.0
    peak = 0.0
    max_drop = 0.0
    for pnl in daily_pnls:
        cumulative += pnl
        peak = max(peak, cumulative)
        max_drop = max(max_drop, peak - cumulative)
    return _ratio(max_drop, half_book_size)


def _price_payload(data: MarketData, selected: list[str], config: BacktestConfig) -> dict[str, list[dict[str, Any]]]:
    dates = data.window_dates(config.start, config.end)
    payload = {}
    for symbol in selected:
        first = next((data.close(day, symbol) for day in dates if data.close(day, symbol)), None)
        if not first:
            continue
        payload[symbol] = [
            {"date": day.isoformat(), "value": (data.close(day, symbol) / first) - 1.0}
            for day in dates
            if data.close(day, symbol) is not None
        ]
    return payload


def _bar_payload(data: MarketData, selected: list[str], config: BacktestConfig) -> dict[str, list[dict[str, Any]]]:
    dates = data.window_dates(config.start, config.end)
    payload = {}
    for symbol in selected:
        rows = []
        for day in dates:
            bar = data.bar(day, symbol)
            if bar is None:
                continue
            rows.append(
                {
                    "date": day.isoformat(),
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                }
            )
        payload[symbol] = rows
    return payload


def _weights_payload(result, selected: list[str]) -> dict[str, list[dict[str, Any]]]:
    return {
        symbol: [
            {"date": point.date.isoformat(), "value": point.weights.get(symbol, 0.0)}
            for point in result.equity_curve
        ]
        for symbol in selected
    }


def _stock_summary(data: MarketData, symbol: str, config: BacktestConfig, result, stock_catalog: list[StockInfo]) -> dict[str, Any]:
    info = next((stock for stock in stock_catalog if stock.symbol == symbol), None)
    dates = data.window_dates(config.start, config.end)
    first = next((data.close(day, symbol) for day in dates if data.close(day, symbol)), None)
    last = next((data.close(day, symbol) for day in reversed(dates) if data.close(day, symbol)), None)
    weights = [point.weights.get(symbol, 0.0) for point in result.equity_curve]
    active_days = sum(1 for weight in weights if weight)
    long_days = sum(1 for weight in weights if weight > 0.0)
    short_days = sum(1 for weight in weights if weight < 0.0)
    avg_abs_weight = sum(abs(weight) for weight in weights) / len(weights)
    avg_signed_weight = sum(weights) / len(weights)
    return {
        "symbol": symbol,
        "name": info.name if info else symbol,
        "region": info.region if info else "",
        "source": info.source if info else "",
        "sector": info.sector if info else "",
        "industry": (info.industry or STOCK_INDUSTRIES.get(symbol, "")) if info else "",
        "firstClose": first,
        "lastClose": last,
        "totalReturn": (last / first) - 1.0 if first and last else 0.0,
        "avgAbsWeight": avg_abs_weight,
        "avgSignedWeight": avg_signed_weight,
        "activeDays": active_days,
        "longDays": long_days,
        "shortDays": short_days,
    }


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PortfolioLab Console</title>
  <style>
    :root {
      --bg: #f4f6f8;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #5d6877;
      --line: #d8dee8;
      --accent: #1b4d89;
      --green: #00876c;
      --red: #c84c3f;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      background: var(--bg);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    main {
      max-width: 1680px;
      margin: 0 auto;
      padding: 18px;
    }
    header {
      display: flex;
      justify-content: space-between;
      gap: 18px;
      align-items: end;
      margin-bottom: 14px;
    }
    h1, h2, h3 { margin: 0; letter-spacing: 0; }
    h1 { font-size: 28px; }
    h2 { font-size: 17px; margin-bottom: 10px; }
    h3 { font-size: 13px; margin-bottom: 8px; color: var(--muted); }
    button, select, input, textarea {
      font: inherit;
    }
    button {
      border: 1px solid var(--accent);
      background: var(--accent);
      color: white;
      border-radius: 6px;
      padding: 9px 12px;
      cursor: pointer;
    }
    select, input, textarea {
      border: 1px solid var(--line);
      border-radius: 6px;
      background: white;
      color: var(--ink);
      padding: 8px;
      width: 100%;
    }
    textarea {
      /* Default height for code editors unless a specific editor overrides it below. */
      min-height: 340px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      line-height: 1.45;
      resize: vertical;
    }
    label {
      display: grid;
      gap: 5px;
      color: var(--muted);
      font-size: 12px;
    }
    .layout {
      display: grid;
      /* Main page columns: left control column width, then the right research/code column. */
      grid-template-columns: 460px minmax(0, 1fr);
      /* Horizontal gap between the left and right page columns. */
      gap: 14px;
      align-items: start;
    }
    .panel {
      /* Shared box style for every white card-like work area. */
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      /* Shared inner padding for boxes. */
      padding: 13px;
      overflow-x: auto;
    }
    .compact-title h2 {
      /* Title-to-content gap for compact boxes such as Market Data and Stock Research. */
      margin-bottom: 2px;
    }
    /* Vertical gap between boxes inside stacked columns. */
    .stack { display: grid; gap: 14px; }
    /* Smaller vertical gap used inside compact control boxes. */
    .compact-stack { gap: 6px; }
    /* Gap between Stock Research and Signal Strategy in the right column. */
    .results-stack { gap: 14px; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .three { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; }
    .full { grid-column: 1 / -1; }
    .symbol-input-box {
      display: grid;
      gap: 8px;
    }
    .symbol-input-box textarea {
      /* Height of the Universe symbol input box. */
      min-height: 86px;
      resize: vertical;
    }
    .data-actions {
      display: grid;
      /* Market Data box: Source and History columns. */
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      align-items: end;
    }
    .data-actions button {
      min-width: 108px;
    }
    .strategy-editor-tools {
      display: grid;
      gap: 8px;
    }
    .strategy-layout {
      display: grid;
      /* Data Preprocessing and Strategy Construction layout: controls above code editor. */
      grid-template-columns: 1fr;
      /* Gap between selector/name/buttons and the code editor. */
      gap: 12px;
      align-items: start;
    }
    .strategy-actions {
      display: grid;
      /* Save / Save As / Delete button widths inside code-library boxes. */
      grid-template-columns: repeat(3, minmax(110px, auto));
      gap: 8px;
      align-items: center;
      justify-content: start;
    }
    .strategy-actions button {
      width: auto;
      min-width: 92px;
      padding: 7px 9px;
      font-size: 12px;
    }
    .neutralization-panel textarea {
      /* Height of the Neutralization code editor. */
      min-height: 180px;
    }
    .strategy-panel textarea {
      /* Height of the Strategy Code editor inside Strategy Construction. */
      min-height: 420px;
    }
    #preprocessorCode {
      /* Height of the Data Preprocessing code editor. */
      min-height: 320px;
    }
    button.secondary {
      background: #ffffff;
      color: var(--ink);
      border-color: var(--line);
    }
    button:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }
    .stock-list {
      display: grid;
      row-gap: 7px;
      /* Available Stocks: fixed row height for each ticker box. */
      grid-auto-rows: 64px;
      /* Available Stocks: scrollable table height. */
      height: 325px;
      overflow: auto;
      padding-right: 4px;
    }
    .basket-actions {
      display: grid;
      gap: 6px;
      justify-items: start;
      margin-top: 10px;
    }
    .stock {
      display: grid;
      /* Available Stocks row columns: checkbox, stock text, delete button. */
      grid-template-columns: 22px minmax(0, 1fr) 30px;
      gap: 8px;
      align-items: center;
      /* Available Stocks row height; keep this aligned with .stock-list grid-auto-rows. */
      min-height: 64px;
      height: 64px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
    }
    .stock .meta {
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .stock input[type="checkbox"] { width: 16px; height: 16px; }
    .stock input[type="number"] { padding: 6px; }
    .icon-button {
      width: 28px;
      height: 28px;
      display: inline-grid;
      place-items: center;
      border-radius: 6px;
      border: 1px solid var(--line);
      color: var(--accent);
      background: #ffffff;
      padding: 0;
    }
    .icon-button.active {
      color: white;
      border-color: var(--green);
      background: var(--green);
    }
    .symbol { font-weight: 700; font-size: 13px; }
    .meta, .status { color: var(--muted); font-size: 12px; }
    .results-heading { margin-bottom: 0; }
    .chart-grid {
      display: grid;
      grid-template-columns: 1fr;
      gap: 14px;
    }
    .chart-toolbar {
      display: grid;
      /* Backtesting Results toolbar: plot selector plus flexible empty space. */
      grid-template-columns: minmax(180px, 260px) 1fr;
      gap: 10px;
      align-items: end;
      margin-bottom: 8px;
    }
    .portfolio-chart-panel {
      display: grid;
      /* Inner gap for the Backtesting Results title, selector, chart, and range bar. */
      gap: 8px;
    }
    .portfolio-chart-panel .chart-toolbar {
      margin-bottom: 0;
    }
    .results-grid {
      display: grid;
      /* Bottom area columns: Backtesting Results chart width, then Performance/Selected Stocks width. */
      grid-template-columns: minmax(0, 1.1fr) minmax(520px, 0.95fr);
      gap: 14px;
      align-items: start;
      /* Gap above Backtesting Results and Performance after the Run Backtest row. */
      margin-top: 6px;
    }
    .backtest-action {
      display: grid;
      gap: 6px;
      justify-items: start;
      /* Gap above the Run Backtest button. */
      margin-top: 10px;
    }
    .backtest-action button {
      min-width: 180px;
    }
    .backtest-status {
      /* Keeps a fixed line reserved for backtest success/error messages below the button. */
      min-height: 18px;
      color: var(--muted);
      font-size: 13px;
    }
    .backtest-status.error-text {
      color: var(--red);
    }
    .table-grid {
      display: grid;
      /* Right side of results: Performance box above Selected Stocks box. */
      grid-template-columns: 1fr;
      /* Gap between Performance and Selected Stocks boxes. */
      gap: 10px;
    }
    .compact-results-tables th,
    .compact-results-tables td {
      padding: 5px 6px;
      font-size: 12px;
    }
    svg { display: block; width: 100%; height: auto; }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    th, td {
      padding: 8px 9px;
      border-bottom: 1px solid #edf0f4;
      text-align: right;
      white-space: nowrap;
    }
    th:first-child, td:first-child { text-align: left; }
    th { color: var(--muted); background: #fafbfc; font-weight: 600; }
    .research-panel {
      display: grid;
      /* Stock Research vertical parts: title, stock list, toolbar, chart. */
      grid-template-rows: auto auto auto minmax(0, 1fr);
      gap: 2px;
      /* Overall Stock Research box height. Increase this if the range slider is hidden. */
      height: 910px;
      overflow: hidden;
    }
    .research-toolbar {
      display: grid;
      /* Stock Research controls: plot type, window selector, remaining flexible space. */
      grid-template-columns: minmax(180px, 280px) minmax(130px, 170px) 1fr;
      gap: 10px;
      align-items: end;
      /* Gap above Research Plot / Window controls. */
      margin-top: 10px;
    }
    .run-row {
      display: flex;
      justify-content: flex-start;
    }
    .run-row button {
      min-width: 180px;
    }
    .stock-detail-grid {
      display: grid;
      /* Stock Research detail area uses one column; chart and legend stack vertically. */
      grid-template-columns: 1fr;
      gap: 14px;
      align-items: start;
      min-height: 0;
    }
    .research-stock-list {
      /* Stock Research scrollable stock table height. */
      height: 150px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    .research-list-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
      margin-bottom: 8px;
    }
    .research-list-head h3 {
      margin: 0;
    }
    .research-actions {
      display: flex;
      justify-content: flex-end;
      gap: 7px;
      flex-wrap: wrap;
    }
    .research-actions button {
      width: auto;
      padding: 6px 9px;
      font-size: 12px;
      border-radius: 6px;
    }
    .research-stock-list table {
      border: 0;
      table-layout: fixed;
      /* Minimum width before the Stock Research list scrolls horizontally. */
      min-width: 700px;
    }
    .research-stock-list th,
    .research-stock-list td {
      text-align: left;
      padding: 7px 8px;
    }
    .research-stock-list th {
      position: sticky;
      top: 0;
      z-index: 1;
    }
    .research-action-col { width: 40px; }
    /* Stock Research list column widths. Adjust these to change Symbol/Name/Region/Industry spacing. */
    .research-portfolio-col { width: 78px; }
    .research-symbol-col { width: 82px; }
    .research-name-col { width: 240px; }
    .research-region-col { width: 100px; }
    .research-industry-col { width: 170px; }
    .research-weight-col { width: 86px; }
    .research-stock-list .truncate {
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    #stockDetailChart {
      /* Stock Research chart container. The SVG height is controlled by .chart-frame > svg and drawChart options. */
      min-height: 0;
      overflow: hidden;
      max-width: 100%;
    }
    .chart-frame {
      max-width: 100%;
      overflow: hidden;
    }
    .chart-frame > svg {
      /* Shared chart height, used by Stock Research unless a more specific rule overrides it. */
      height: 480px;
      max-height: 480px;
      width: 100%;
      max-width: 100%;
    }
    #portfolioChart .chart-frame > svg {
      /* Displayed height for Backtesting Results charts only. */
      height: 370px;
      max-height: 370px;
    }
    .chart-legend-viewport {
      /* Horizontal scrolling strip for chart legends when many stocks are displayed. */
      max-width: 100%;
      min-width: 0;
      overflow-x: auto;
      overflow-y: hidden;
      scrollbar-gutter: stable;
      scrollbar-width: none;
    }
    .chart-legend-viewport::-webkit-scrollbar {
      display: none;
      width: 0;
      height: 0;
    }
    .chart-legend-viewport::-webkit-scrollbar-track {
      background: transparent;
    }
    .chart-legend-viewport::-webkit-scrollbar-thumb {
      background: #aab4c2;
      border-radius: 999px;
    }
    .chart-legend-viewport:hover::-webkit-scrollbar,
    .chart-legend-viewport:active::-webkit-scrollbar {
      height: 0;
    }
    .chart-legend-viewport:hover::-webkit-scrollbar-thumb,
    .chart-legend-viewport:active::-webkit-scrollbar-thumb {
      background: #aab4c2;
    }
    .chart-legend-strip {
      display: flex;
      gap: 14px;
      align-items: center;
      width: max-content;
      min-width: 100%;
      justify-content: center;
      white-space: nowrap;
      padding: 7px 0 2px;
      margin-top: -2px;
      flex-wrap: nowrap;
    }
    .legend-item {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      color: var(--ink);
      font-size: 11px;
      flex: 0 0 auto;
    }
    .legend-swatch {
      width: 18px;
      height: 3px;
      border-radius: 999px;
    }
    .range-zoom {
      /* Draggable date-range bar under Stock Research and Backtesting Results charts. */
      border-top: 1px solid var(--line);
      margin-top: 8px;
      padding-top: 10px;
    }
    .range-head {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      justify-content: space-between;
      align-items: end;
      gap: 10px;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 6px;
    }
    .range-date-inputs {
      display: grid;
      grid-template-columns: minmax(130px, 1fr) minmax(130px, 1fr);
      gap: 8px;
      min-width: 0;
    }
    .range-date-inputs input {
      padding: 5px 7px;
      font-size: 12px;
    }
    .range-reset {
      align-self: end;
      background: #ffffff;
      color: var(--ink);
      border: 1px solid var(--line);
      border-radius: 6px;
      height: 30px;
      padding: 5px 9px;
      width: auto;
      font-size: 12px;
    }
    .range-track {
      position: relative;
      /* Height of the draggable range slider track. */
      height: 28px;
    }
    .range-track-line,
    .range-track-fill {
      position: absolute;
      left: 0;
      right: 0;
      top: 13px;
      height: 4px;
      border-radius: 999px;
    }
    .range-track-line {
      background: #e1e6ed;
    }
    .range-track-fill {
      background: #1b4d89;
    }
    .range-track input[type="range"] {
      position: absolute;
      left: 0;
      top: 0;
      width: 100%;
      height: 28px;
      margin: 0;
      background: transparent;
      pointer-events: none;
      appearance: none;
    }
    .range-track input[type="range"]::-webkit-slider-thumb {
      appearance: none;
      width: 14px;
      height: 14px;
      border-radius: 999px;
      background: #ffffff;
      border: 2px solid #1b4d89;
      box-shadow: 0 1px 3px rgba(23, 32, 42, 0.25);
      pointer-events: auto;
      cursor: ew-resize;
    }
    .range-track input[type="range"]::-moz-range-thumb {
      width: 14px;
      height: 14px;
      border-radius: 999px;
      background: #ffffff;
      border: 2px solid #1b4d89;
      box-shadow: 0 1px 3px rgba(23, 32, 42, 0.25);
      pointer-events: auto;
      cursor: ew-resize;
    }
    .detail-facts {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }
    .fact {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px;
      background: #fbfcfd;
    }
    .fact span { display: block; color: var(--muted); font-size: 12px; }
    .fact strong { font-size: 15px; }
    .metric-groups {
      display: grid;
      /* Performance box: Aggregate / Training / Testing metric columns. */
      grid-template-columns: repeat(3, minmax(180px, 1fr));
      gap: 12px;
      margin-bottom: 10px;
    }
    .metric-group {
      border: 0;
      border-radius: 0;
      padding: 0;
      background: transparent;
    }
    .metric-grid {
      display: grid;
      gap: 4px;
    }
    .metric-grid .fact {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 8px;
      align-items: baseline;
      border: 0;
      padding: 0;
      background: transparent;
    }
    .metric-grid .fact strong { font-size: 13px; }
    @media (max-width: 1080px) {
      header, .layout, .chart-grid, .chart-toolbar, .research-toolbar, .table-grid, .results-grid, .strategy-layout, .row, .three, .stock-detail-grid, .metric-groups {
        grid-template-columns: 1fr;
      }
      .research-panel {
        height: auto;
        overflow: visible;
      }
      .stock-list {
        height: 320px;
      }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>PortfolioLab Console</h1>
        <div class="status" id="periodStatus"></div>
      </div>
    </header>

    <section class="layout">
      <aside class="stack">
        <!-- Box: Market Data. Source/history controls and loaded-data status. Sizing mainly uses .panel, .compact-stack, and .data-actions. -->
        <div class="panel stack compact-stack compact-title">
          <h2>Market Data</h2>
          <!-- Market Data controls: source selector and history length selector. -->
          <div class="data-actions">
            <label>Source
              <select id="dataSource">
                <option value="yahoo">Yahoo Finance / yfinance</option>
              </select>
            </label>
            <label>History
              <select id="dataPeriod">
                <option value="5y">5Y</option>
                <option value="10y" selected>10Y</option>
                <option value="max">Max</option>
              </select>
            </label>
          </div>
          <!-- Market Data status line. -->
          <div class="status" id="dataStatus"></div>
        </div>

        <!-- Box: Universe. Manual ticker lookup input. Textarea height is controlled by .symbol-input-box textarea. -->
        <div class="panel universe-panel compact-title">
          <h2>Universe</h2>
          <div class="symbol-input-box">
            <label>Search Symbols
              <textarea id="symbolLookupInput" placeholder="AAPL, MSFT, BRK-B, MC.PA"></textarea>
            </label>
            <button id="addSymbolsButton" type="button">Search And Add</button>
            <div class="status" id="lookupStatus"></div>
          </div>
        </div>

        <!-- Box: Available Stocks. Scrollable ticker list. Box rows use .stock-list height and .stock grid sizing. -->
        <div class="panel weights-panel">
          <h2 id="stocksPanelTitle">Available Stocks</h2>
          <!-- Available Stocks list rendered by renderStocks(). -->
          <div id="stocks" class="stock-list"></div>
          <!-- Available Stocks action area: refresh prices for all available tickers. -->
          <div class="basket-actions">
            <button id="loadPriceDataButton" type="button">Refresh Price Data</button>
            <div class="status" id="priceDataStatus"></div>
          </div>
        </div>

        <!-- Box: Backtest Setup. Dates, train/test split, rebalance, portfolio mode, and trading-cost inputs. -->
        <div class="panel stack compact-stack compact-title">
          <h2>Backtest Setup</h2>
          <!-- Backtest Setup date inputs. -->
          <div class="row">
            <label>Start <input id="start" type="date"></label>
            <label>End <input id="end" type="date"></label>
          </div>
          <!-- Backtest Setup split and rebalance inputs. -->
          <div class="row">
            <label>Train/Test Split <input id="splitDate" type="date"></label>
            <label>Rebalance
              <select id="rebalanceFrequency">
                <option value="monthly">Monthly</option>
                <option value="weekly">Weekly</option>
                <option value="daily">Daily</option>
              </select>
            </label>
          </div>
          <!-- Backtest Setup portfolio weighting and cost inputs. -->
          <div class="three">
            <label class="full">Portfolio Weights
              <select id="portfolioMode">
                <option value="equal">Equal Weight</option>
                <option value="manual">Manual Weights</option>
              </select>
            </label>
            <label>Cost bps <input id="transactionCostBps" type="number" value="5" step="0.5"></label>
            <label>Slippage bps <input id="slippageBps" type="number" value="2" step="0.5"></label>
          </div>
        </div>

        <!-- Box: Neutralization. Saved grouping/neutralization code library. Code editor height uses .neutralization-panel textarea. -->
        <div class="panel stack compact-stack compact-title neutralization-panel">
          <h2>Neutralization</h2>
          <!-- Neutralization selector: built-in or saved custom grouping method. -->
          <label>Neutralization
            <select id="neutralizerSelect" aria-label="Neutralization code"></select>
          </label>
          <!-- Neutralization library controls: name and save/delete buttons. -->
          <label>Name <input id="neutralizerName" type="text" placeholder="Market neutralization"></label>
          <div class="strategy-actions">
            <button id="saveNeutralizerButton" type="button">Save / Update</button>
            <button id="saveNeutralizerAsButton" type="button">Save As New</button>
            <button id="deleteNeutralizerButton" class="secondary" type="button">Delete</button>
          </div>
          <div class="status" id="neutralizerStatus"></div>
          <!-- Neutralization code editor. -->
          <label>Grouping And Neutralization Code <textarea id="neutralizerCode"></textarea></label>
        </div>

      </aside>

      <section class="stack results-stack">
        <!-- Box: Stock Research. Stock table, plot controls, chart, legend, and range slider. Overall height uses .research-panel height. -->
        <div class="panel research-panel compact-title">
          <h2>Stock Research</h2>
          <!-- Stock Research stock/portfolio/weight table rendered by renderStockDetail(). List height uses .research-stock-list. -->
          <div id="stockResearchList"></div>
          <!-- Stock Research chart selectors. Top gap uses .research-toolbar margin-top. -->
          <div class="research-toolbar">
            <label>Research Plot
              <select id="researchChartType">
                <option value="raw">Price</option>
                <option value="candle_day">Candlestick: Day</option>
                <option value="candle_week">Candlestick: Week</option>
                <option value="candle_quarter">Candlestick: Quarter</option>
                <option value="candle_year">Candlestick: Year</option>
              </select>
            </label>
            <label>Window
              <select id="researchWindow">
                <option value="all">All</option>
                <option value="63">3M</option>
                <option value="126">6M</option>
                <option value="252" selected>1Y</option>
                <option value="756">3Y</option>
                <option value="custom">Custom</option>
              </select>
            </label>
          </div>
          <!-- Stock Research figure container. Chart height uses drawChart height and .chart-frame > svg. -->
          <div id="stockDetailChart"></div>
        </div>
        <!-- Box: Data Preprocessing. Saved preprocessing code library. Code editor height uses #preprocessorCode. -->
        <div id="preprocessingBox" class="panel preprocessing-panel compact-title">
          <h2>Data Preprocessing</h2>
          <div class="strategy-layout">
            <!-- Data Preprocessing library selector, name, buttons, and status. -->
            <div class="strategy-editor-tools">
              <label>Saved Preprocessing
                <select id="dataPreprocessor" aria-label="Data preprocessing"></select>
              </label>
              <label>Preprocessing Name <input id="preprocessorName" type="text" placeholder="Backfill missing bars"></label>
              <div class="strategy-actions">
                <button id="savePreprocessorButton" type="button">Save / Update</button>
                <button id="savePreprocessorAsButton" type="button">Save As New</button>
                <button id="deletePreprocessorButton" class="secondary" type="button">Delete</button>
              </div>
              <div class="status" id="preprocessorStatus"></div>
            </div>
            <!-- Data Preprocessing code editor. -->
            <label>Preprocessing Code <textarea id="preprocessorCode"></textarea></label>
          </div>
        </div>
        <!-- Box: Strategy Construction. Saved strategy code library. Code editor height uses .strategy-panel textarea. -->
        <div id="codeBox" class="panel strategy-panel compact-title">
          <h2>Strategy Construction</h2>
          <div class="strategy-layout">
            <!-- Strategy Construction library selector, name, buttons, and status. -->
            <div class="strategy-editor-tools">
              <label>Saved Strategy
                <select id="signalStrategy" aria-label="Signal strategy"></select>
              </label>
              <label>Strategy Name <input id="strategyName" type="text" placeholder="My momentum signal"></label>
              <div class="strategy-actions">
                <button id="saveStrategyButton" type="button">Save / Update</button>
                <button id="saveStrategyAsButton" type="button">Save As New</button>
                <button id="deleteStrategyButton" class="secondary" type="button">Delete</button>
              </div>
              <div class="status" id="strategyStatus"></div>
            </div>
            <!-- Strategy Construction code editor. -->
            <label>Strategy Code <textarea id="code"></textarea></label>
          </div>
        </div>
      </section>
    </section>
    <!-- Box/row: Run Backtest action. Gap above button uses .backtest-action margin-top. -->
    <section class="backtest-action">
      <button id="runButton" type="button">Run Backtest</button>
      <!-- Backtest status and Python/code error messages. Reserved line height uses .backtest-status min-height. -->
      <div class="backtest-status" id="backtestStatus"></div>
    </section>
    <!-- Bottom result area. Column widths use .results-grid grid-template-columns. -->
    <section class="results-grid">
      <!-- Box: Backtesting Results. Plot selector, chart, legend, and date-range slider. Chart height uses #portfolioChart .chart-frame > svg and portfolioChartOptions. -->
      <div class="panel portfolio-chart-panel">
        <h2 class="results-heading" id="summary">Backtesting Results</h2>
        <!-- Backtesting Results plot selector. -->
        <div class="chart-toolbar">
          <label>Plot
            <select id="portfolioChartType">
              <option value="pnl">Portfolio PnL</option>
              <option value="prices">Normalized Prices</option>
              <option value="weights">Signal Weights</option>
              <option value="drawdown">Drawdown</option>
              <option value="sharpe">Sharpe</option>
              <option value="turnover">Turnover</option>
            </select>
          </label>
        </div>
        <!-- Backtesting Results figure container rendered by renderPortfolioChart(). -->
        <div id="portfolioChart"></div>
      </div>
      <!-- Right result column. Vertical gap between boxes uses .table-grid gap. -->
      <div class="table-grid compact-results-tables">
        <!-- Box: Performance. Aggregate metric groups plus yearly table. Table density uses .compact-results-tables. -->
        <div class="panel">
          <h2>Performance</h2>
          <div id="metricsTable"></div>
        </div>
        <!-- Box: Selected Stocks. Per-stock return/weight/activity table. Table density uses .compact-results-tables. -->
        <div class="panel">
          <h2>Selected Stocks</h2>
          <div id="stocksTable"></div>
        </div>
      </div>
    </section>
  </main>

  <script>
    const colors = ["#1b4d89", "#00876c", "#c84c3f", "#7f5ab6", "#b88700", "#2f6f73", "#4059ad", "#7b6d8d"];
    const state = {
      stocks: [],
      rawPrices: {},
      rawBars: {},
      pools: [],
      preprocessorLibrary: [],
      strategyLibrary: [],
      neutralizerLibrary: [],
      researchSelected: new Set(),
      selected: new Set(),
      weights: {},
      loadedSymbols: new Set(),
      detailSymbol: null,
      compareSymbols: [],
      researchSelectionCleared: false,
      researchRange: null,
      portfolioRange: null,
      currentResult: null
    };

    const $ = (id) => document.getElementById(id);
    const pct = (value) => `${(value * 100).toFixed(2)}%`;
    const bps = (value) => `${(value * 10000).toFixed(2)} bps`;
    const num = (value) => Number(value || 0).toLocaleString(undefined, { maximumFractionDigits: 2 });
    const escapeHtml = (value) => String(value).replace(/[&<>"']/g, (char) => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;"
    }[char]));

    async function init() {
      bindEvents();
      await loadPreprocessorLibrary("none");
      await loadStrategyLibrary("none");
      await loadNeutralizerLibrary("none");
      const response = await fetch("/api/pools");
      const catalog = await response.json();
      if (!response.ok) {
        $("dataStatus").textContent = catalog.error || "Market data is not loaded";
        setBacktestStatus("Load market data before running a backtest.", true);
        return;
      }
      applyCatalog(catalog, false);
      if (state.loadedSymbols.size && state.selected.size) {
        await runBacktest();
      } else {
        setBacktestStatus("Search symbols, choose portfolio names, then load price data before running a backtest.");
      }
    }

    function applyCatalog(catalog, preserveSelection) {
      // Page initialization: fills Market Data status, Available Stocks, Stock Research defaults, and date inputs.
      const previousSelected = preserveSelection ? new Set(state.selected) : null;
      const previousResearchSelected = preserveSelection ? new Set(state.researchSelected) : null;
      const previousWeights = preserveSelection ? { ...state.weights } : null;
      const previousStocks = preserveSelection ? state.stocks : [];
      state.rawPrices = catalog.rawPrices || {};
      state.rawBars = catalog.rawBars || {};
      state.pools = catalog.pools;
      state.loadedSymbols = new Set(catalog.data?.symbols || []);
      const previousAvailable = new Set(previousStocks.map((stock) => stock.symbol));
      const incomingStocks = (catalog.stocks || []).filter((stock) => previousAvailable.has(stock.symbol) || stock.loaded || state.loadedSymbols.has(stock.symbol));
      state.stocks = mergeStocks(previousStocks, incomingStocks);
      const defaults = catalog.defaults;
      $("start").value = defaults.start;
      $("end").value = defaults.end;
      $("splitDate").value = defaults.splitDate;
      $("rebalanceFrequency").value = defaults.rebalanceFrequency;
      syncStrategyEditor();
      const available = new Set(state.stocks.map((stock) => stock.symbol));
      state.researchSelected = preserveSelection
        ? new Set(Array.from(previousResearchSelected).filter((symbol) => available.has(symbol)))
        : new Set(state.stocks.filter((stock) => stock.loaded).map((stock) => stock.symbol));
      state.selected = preserveSelection
        ? new Set(Array.from(previousSelected).filter((symbol) => state.researchSelected.has(symbol)))
        : new Set();
      state.weights = {};
      Array.from(state.selected).forEach((symbol) => {
        state.weights[symbol] = previousWeights?.[symbol] ?? +(100 / Math.max(1, state.selected.size)).toFixed(2);
      });
      state.detailSymbol = Array.from(state.researchSelected)[0] || state.stocks[0]?.symbol || null;
      state.compareSymbols = state.compareSymbols.filter((symbol) => available.has(symbol));
      if (!state.compareSymbols.length && state.detailSymbol) {
        state.compareSymbols = [state.detailSymbol];
      }
      state.researchSelectionCleared = false;
      state.researchRange = null;
      const loadedText = catalog.data?.loadedCount
        ? `${catalog.data.loadedCount} loaded · ${catalog.data.start || ""} to ${catalog.data.end || ""}`
        : `${state.stocks.length} available symbols · prices not loaded`;
      $("dataStatus").textContent = `${catalog.data?.source || "Data"} · ${loadedText}`;
      renderStocks();
      renderStockDetail();
    }

    function mergeStocks(base, additions) {
      const merged = new Map();
      [...base, ...additions].forEach((stock) => {
        if (!stock?.symbol) return;
        const current = merged.get(stock.symbol) || {};
        merged.set(stock.symbol, { ...current, ...stock });
      });
      return Array.from(merged.values()).sort((left, right) => left.symbol.localeCompare(right.symbol));
    }

    function bindEvents() {
      $("runButton").addEventListener("click", runBacktest);
      $("addSymbolsButton").addEventListener("click", addSymbols);
      $("loadPriceDataButton").addEventListener("click", loadPriceData);
      $("researchChartType").addEventListener("change", () => {
        renderStockDetail();
      });
      $("researchWindow").addEventListener("change", () => {
        applyResearchWindowPreset(currentResearchStocks(), false);
        renderStockDetail();
      });
      $("portfolioChartType").addEventListener("change", () => {
        renderPortfolioChart();
      });
      $("dataPreprocessor").addEventListener("change", syncPreprocessorEditor);
      $("savePreprocessorButton").addEventListener("click", () => {
        savePreprocessor(false);
      });
      $("savePreprocessorAsButton").addEventListener("click", () => {
        savePreprocessor(true);
      });
      $("deletePreprocessorButton").addEventListener("click", deleteSelectedPreprocessor);
      $("signalStrategy").addEventListener("change", () => {
        syncStrategyEditor();
      });
      $("saveStrategyButton").addEventListener("click", () => {
        saveStrategy(false);
      });
      $("saveStrategyAsButton").addEventListener("click", () => {
        saveStrategy(true);
      });
      $("deleteStrategyButton").addEventListener("click", deleteSelectedStrategy);
      $("neutralizerSelect").addEventListener("change", syncNeutralizerEditor);
      $("saveNeutralizerButton").addEventListener("click", () => {
        saveNeutralizer(false);
      });
      $("saveNeutralizerAsButton").addEventListener("click", () => {
        saveNeutralizer(true);
      });
      $("deleteNeutralizerButton").addEventListener("click", deleteSelectedNeutralizer);
    }

    async function loadPreprocessorLibrary(selectedId) {
      // Data Preprocessing box: loads saved preprocessing choices into the selector.
      const response = await fetch("/api/preprocessors");
      const result = await response.json();
      if (!response.ok) {
        setPreprocessorStatus(result.error || "Preprocessing library could not be loaded.");
        return;
      }
      state.preprocessorLibrary = result.preprocessors || [];
      renderPreprocessorLibrary(selectedId || state.preprocessorLibrary[0]?.id);
      setPreprocessorStatus("");
    }

    async function postPreprocessorLibrary(url, payload) {
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const result = await response.json();
      if (!response.ok) {
        throw new Error(result.error || "Preprocessing library update failed.");
      }
      state.preprocessorLibrary = result.preprocessors || [];
      renderPreprocessorLibrary(result.selectedId);
      setPreprocessorStatus(result.message);
      return result;
    }

    function preprocessorNameExists(name, exceptId) {
      return state.preprocessorLibrary.some((preprocessor) =>
        preprocessor.name.toLowerCase() === name.toLowerCase() && preprocessor.id !== exceptId
      );
    }

    function selectedPreprocessor() {
      return state.preprocessorLibrary.find((preprocessor) => preprocessor.id === $("dataPreprocessor").value) || state.preprocessorLibrary[0];
    }

    function renderPreprocessorLibrary(selectedId) {
      const select = $("dataPreprocessor");
      select.innerHTML = state.preprocessorLibrary
        .map((preprocessor) => `<option value="${escapeHtml(preprocessor.id)}">${escapeHtml(preprocessor.name)}</option>`)
        .join("");
      if (selectedId && state.preprocessorLibrary.some((preprocessor) => preprocessor.id === selectedId)) {
        select.value = selectedId;
      } else if (state.preprocessorLibrary.length) {
        select.value = state.preprocessorLibrary[0].id;
      }
      syncPreprocessorEditor();
    }

    function syncPreprocessorEditor() {
      // Data Preprocessing box: copies the selected saved item into name/code fields.
      const preprocessor = selectedPreprocessor();
      $("preprocessorName").value = preprocessor?.name || "";
      $("preprocessorCode").value = preprocessor?.code || "";
      $("deletePreprocessorButton").disabled = state.preprocessorLibrary.length <= 1;
    }

    function setPreprocessorStatus(message) {
      $("preprocessorStatus").textContent = message;
    }

    async function savePreprocessor(saveAsNew) {
      // Data Preprocessing box: save/update/delete validation is handled here before calling the backend.
      const current = selectedPreprocessor();
      const name = $("preprocessorName").value.trim();
      const code = $("preprocessorCode").value.trim();
      if (!name) {
        setPreprocessorStatus("Name is required.");
        return;
      }
      if (!code) {
        setPreprocessorStatus("Preprocessing code is required.");
        return;
      }
      if (preprocessorNameExists(name, saveAsNew ? null : current?.id)) {
        setPreprocessorStatus(`Preprocessing name already exists: ${name}. Please choose a new name.`);
        return;
      }
      try {
        await postPreprocessorLibrary("/api/preprocessors/save", {
          id: current?.id,
          name,
          code,
          saveAsNew
        });
      } catch (error) {
        setPreprocessorStatus(error.message);
      }
    }

    async function deleteSelectedPreprocessor() {
      const preprocessor = selectedPreprocessor();
      if (!preprocessor || state.preprocessorLibrary.length <= 1) {
        setPreprocessorStatus("Keep at least one preprocessor.");
        return;
      }
      try {
        await postPreprocessorLibrary("/api/preprocessors/delete", { id: preprocessor.id });
      } catch (error) {
        setPreprocessorStatus(error.message);
      }
    }

    async function loadStrategyLibrary(selectedId) {
      // Strategy Construction box: loads saved strategy choices into the selector.
      const response = await fetch("/api/strategies");
      const result = await response.json();
      if (!response.ok) {
        setStrategyStatus(result.error || "Strategy library could not be loaded.");
        return;
      }
      state.strategyLibrary = result.strategies || [];
      renderStrategyLibrary(selectedId || state.strategyLibrary[0]?.id);
      setStrategyStatus(result.path ? `Stored in ${result.path}` : "");
    }

    async function postStrategyLibrary(url, payload) {
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const result = await response.json();
      if (!response.ok) {
        throw new Error(result.error || "Strategy library update failed.");
      }
      state.strategyLibrary = result.strategies || [];
      renderStrategyLibrary(result.selectedId);
      setStrategyStatus(result.path ? `${result.message} Stored in ${result.path}` : result.message);
      return result;
    }

    function strategyNameExists(name, exceptId) {
      return state.strategyLibrary.some((strategy) =>
        strategy.name.toLowerCase() === name.toLowerCase() && strategy.id !== exceptId
      );
    }

    function selectedStrategy() {
      return state.strategyLibrary.find((strategy) => strategy.id === $("signalStrategy").value) || state.strategyLibrary[0];
    }

    function renderStrategyLibrary(selectedId) {
      const select = $("signalStrategy");
      select.innerHTML = state.strategyLibrary
        .map((strategy) => `<option value="${escapeHtml(strategy.id)}">${escapeHtml(strategy.name)}</option>`)
        .join("");
      if (selectedId && state.strategyLibrary.some((strategy) => strategy.id === selectedId)) {
        select.value = selectedId;
      } else if (state.strategyLibrary.length) {
        select.value = state.strategyLibrary[0].id;
      }
      syncStrategyEditor();
    }

    function syncStrategyEditor() {
      // Strategy Construction box: copies the selected saved strategy into name/code fields.
      const strategy = selectedStrategy();
      $("strategyName").value = strategy?.name || "";
      $("code").value = strategy?.code || "";
      $("code").readOnly = false;
      $("deleteStrategyButton").disabled = state.strategyLibrary.length <= 1;
    }

    function setStrategyStatus(message) {
      $("strategyStatus").textContent = message;
    }

    function setBacktestStatus(message, isError = false) {
      $("backtestStatus").textContent = message;
      $("backtestStatus").classList.toggle("error-text", isError);
    }

    async function saveStrategy(saveAsNew) {
      // Strategy Construction box: save/update/delete validation is handled here before calling the backend.
      const current = selectedStrategy();
      const name = $("strategyName").value.trim();
      const code = $("code").value.trim();
      if (!name) {
        setStrategyStatus("Name is required.");
        return;
      }
      if (!code) {
        setStrategyStatus("Strategy code is required.");
        return;
      }
      if (strategyNameExists(name, saveAsNew ? null : current?.id)) {
        setStrategyStatus(`Strategy name already exists: ${name}. Please choose a new name.`);
        return;
      }
      try {
        await postStrategyLibrary("/api/strategies/save", {
          id: current?.id,
          name,
          code,
          saveAsNew
        });
      } catch (error) {
        setStrategyStatus(error.message);
      }
    }

    async function deleteSelectedStrategy() {
      const strategy = selectedStrategy();
      if (!strategy || state.strategyLibrary.length <= 1) {
        setStrategyStatus("Keep at least one strategy.");
        return;
      }
      try {
        await postStrategyLibrary("/api/strategies/delete", { id: strategy.id });
      } catch (error) {
        setStrategyStatus(error.message);
      }
    }

    async function loadNeutralizerLibrary(selectedId) {
      // Neutralization box: loads saved grouping/neutralization choices into the selector.
      const response = await fetch("/api/neutralizers");
      const result = await response.json();
      if (!response.ok) {
        setNeutralizerStatus(result.error || "Neutralization library could not be loaded.");
        return;
      }
      state.neutralizerLibrary = result.neutralizers || [];
      renderNeutralizerLibrary(selectedId || state.neutralizerLibrary[0]?.id);
      setNeutralizerStatus("");
    }

    async function postNeutralizerLibrary(url, payload) {
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const result = await response.json();
      if (!response.ok) {
        throw new Error(result.error || "Neutralization library update failed.");
      }
      state.neutralizerLibrary = result.neutralizers || [];
      renderNeutralizerLibrary(result.selectedId);
      setNeutralizerStatus(result.message);
      return result;
    }

    function neutralizerNameExists(name, exceptId) {
      return state.neutralizerLibrary.some((neutralizer) =>
        neutralizer.name.toLowerCase() === name.toLowerCase() && neutralizer.id !== exceptId
      );
    }

    function selectedNeutralizer() {
      return state.neutralizerLibrary.find((neutralizer) => neutralizer.id === $("neutralizerSelect").value) || state.neutralizerLibrary[0];
    }

    function selectedNeutralizationMode() {
      const neutralizer = selectedNeutralizer();
      if (["none", "market", "region", "sector", "industry"].includes(neutralizer?.id)) {
        return neutralizer.id;
      }
      return "market";
    }

    function renderNeutralizerLibrary(selectedId) {
      const select = $("neutralizerSelect");
      select.innerHTML = state.neutralizerLibrary
        .map((neutralizer) => `<option value="${escapeHtml(neutralizer.id)}">${escapeHtml(neutralizer.name)}</option>`)
        .join("");
      if (selectedId && state.neutralizerLibrary.some((neutralizer) => neutralizer.id === selectedId)) {
        select.value = selectedId;
      } else if (state.neutralizerLibrary.length) {
        select.value = state.neutralizerLibrary[0].id;
      }
      syncNeutralizerEditor();
    }

    function syncNeutralizerEditor() {
      // Neutralization box: copies the selected saved item into name/code fields.
      const neutralizer = selectedNeutralizer();
      $("neutralizerName").value = neutralizer?.name || "";
      $("neutralizerCode").value = neutralizer?.code || "";
      $("deleteNeutralizerButton").disabled = state.neutralizerLibrary.length <= 1;
    }

    function setNeutralizerStatus(message) {
      $("neutralizerStatus").textContent = message;
    }

    async function saveNeutralizer(saveAsNew) {
      // Neutralization box: save/update/delete validation is handled here before calling the backend.
      const current = selectedNeutralizer();
      const name = $("neutralizerName").value.trim();
      const code = $("neutralizerCode").value.trim();
      if (!name) {
        setNeutralizerStatus("Name is required.");
        return;
      }
      if (!code) {
        setNeutralizerStatus("Neutralization code is required.");
        return;
      }
      if (neutralizerNameExists(name, saveAsNew ? null : current?.id)) {
        setNeutralizerStatus(`Neutralization name already exists: ${name}. Please choose a new name.`);
        return;
      }
      try {
        await postNeutralizerLibrary("/api/neutralizers/save", {
          id: current?.id,
          name,
          code,
          saveAsNew
        });
      } catch (error) {
        setNeutralizerStatus(error.message);
      }
    }

    async function deleteSelectedNeutralizer() {
      const neutralizer = selectedNeutralizer();
      if (!neutralizer || state.neutralizerLibrary.length <= 1) {
        setNeutralizerStatus("Keep at least one neutralizer.");
        return;
      }
      try {
        await postNeutralizerLibrary("/api/neutralizers/delete", { id: neutralizer.id });
      } catch (error) {
        setNeutralizerStatus(error.message);
      }
    }

    async function addSymbols() {
      // Universe box: searches yfinance-compatible symbols and adds them to Available Stocks.
      const rawSymbols = $("symbolLookupInput").value;
      $("lookupStatus").textContent = "Searching Yahoo Finance...";
      $("addSymbolsButton").disabled = true;
      const payload = {
        source: $("dataSource").value,
        symbols: rawSymbols
      };
      try {
        const response = await fetch("/api/lookup-symbols", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        const result = await response.json();
        if (!response.ok) {
          $("lookupStatus").textContent = result.error || "Symbol search failed";
          return;
        }
        const before = new Set(state.stocks.map((stock) => stock.symbol));
        state.stocks = mergeStocks(state.stocks, result.stocks || []);
        (result.stocks || []).forEach((stock) => {
          state.researchSelected.add(stock.symbol);
          if (!before.has(stock.symbol)) {
            state.weights[stock.symbol] = state.weights[stock.symbol] || 0;
          }
        });
        $("symbolLookupInput").value = "";
        $("lookupStatus").textContent = result.message || "Symbols added";
        renderStocks();
        renderStockDetail();
      } finally {
        $("addSymbolsButton").disabled = false;
      }
    }

    async function loadPriceData() {
      // Available Stocks box: refreshes price data for every ticker still in the available list.
      const selected = state.stocks.map((stock) => stock.symbol);
      if (!selected.length) {
        $("priceDataStatus").textContent = "Add at least one available stock first.";
        return;
      }
      $("priceDataStatus").textContent = `Refreshing prices for ${selected.length} available symbols...`;
      $("loadPriceDataButton").disabled = true;
      const payload = {
        source: $("dataSource").value,
        period: $("dataPeriod").value,
        symbols: selected,
        stocks: state.stocks
      };
      try {
        const response = await fetch("/api/load-price-data", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        const result = await response.json();
        if (!response.ok) {
          $("priceDataStatus").textContent = result.error || "Price data load failed";
          return;
        }
        applyCatalog(result.catalog, true);
        $("priceDataStatus").textContent = result.message || "Price data refreshed";
        if (state.selected.size) {
          await runBacktest();
        }
      } finally {
        $("loadPriceDataButton").disabled = false;
      }
    }

    function visibleStocks() {
      return state.stocks;
    }

    function portfolioStocks() {
      return state.stocks.filter((stock) => state.researchSelected.has(stock.symbol));
    }

    function matchesCatalogSearch(stock, query) {
      if (!query) return true;
      return [stock.symbol, stock.name, stock.region, stock.sector, stock.industry, stock.exchange]
        .some((value) => String(value || "").toLowerCase().includes(query));
    }

    function stockIndustry(stock) {
      return stock.industry || stock.sector || "Industry n/a";
    }

    function renderStocks() {
      // Available Stocks box: renders ticker rows, include-for-research checkbox, and delete button.
      const stocks = visibleStocks();
      $("stocksPanelTitle").textContent = `Available Stocks (${stocks.length} all · ${state.researchSelected.size} research)`;
      if (!stocks.length) {
        $("stocks").innerHTML = `<div class="meta">Search and add tickers in Universe first.</div>`;
        return;
      }
      $("stocks").innerHTML = stocks.map((stock) => {
        const checked = state.researchSelected.has(stock.symbol);
        const loaded = state.loadedSymbols.has(stock.symbol);
        return `
          <div class="stock">
            <input type="checkbox" data-available-symbol="${stock.symbol}" ${checked ? "checked" : ""} title="Include ${stock.symbol} in Stock Research">
            <div>
              <div class="symbol">${stock.symbol} · ${stock.name}</div>
              <div class="meta">${stock.region || "Region n/a"} · ${stockIndustry(stock)} · ${stock.exchange || "Exchange n/a"} · ${loaded ? "loaded" : "not loaded"}</div>
            </div>
            <button class="icon-button" data-remove-available="${stock.symbol}" title="Delete ${stock.symbol} from Available Stocks" aria-label="Delete ${stock.symbol} from Available Stocks">x</button>
          </div>
        `;
      }).join("");
      document.querySelectorAll("[data-available-symbol]").forEach((node) => {
        node.addEventListener("change", () => {
          const symbol = node.dataset.availableSymbol;
          if (node.checked) {
            state.researchSelected.add(symbol);
            state.weights[symbol] = state.weights[symbol] || 0;
            state.detailSymbol = symbol;
          } else {
            state.researchSelected.delete(symbol);
            state.selected.delete(symbol);
            state.weights[symbol] = 0;
            state.compareSymbols = state.compareSymbols.filter((item) => item !== symbol);
          }
          renderStocks();
          renderStockDetail();
        });
      });
      document.querySelectorAll("[data-remove-available]").forEach((node) => {
        node.addEventListener("click", () => {
          removeAvailableStock(node.dataset.removeAvailable);
        });
      });
    }

    function removeAvailableStock(symbol) {
      // Available Stocks box: removes one ticker from research, portfolio, cached bars, and the visible list.
      state.stocks = state.stocks.filter((stock) => stock.symbol !== symbol);
      state.researchSelected.delete(symbol);
      state.selected.delete(symbol);
      state.loadedSymbols.delete(symbol);
      delete state.weights[symbol];
      delete state.rawPrices[symbol];
      delete state.rawBars[symbol];
      state.compareSymbols = state.compareSymbols.filter((item) => item !== symbol);
      if (state.detailSymbol === symbol) {
        state.detailSymbol = Array.from(state.researchSelected)[0] || state.stocks[0]?.symbol || null;
      }
      $("priceDataStatus").textContent = `${symbol} removed. Refresh price data to update the local cache.`;
      renderStocks();
      renderStockDetail();
    }

    async function runBacktest() {
      // Run Backtest row: builds the payload from setup/code boxes and sends it to the backtest API.
      setBacktestStatus("Running backtest...");
      $("runButton").disabled = true;
      const selected = Array.from(state.selected);
      const unloaded = selected.filter((symbol) => !state.loadedSymbols.has(symbol));
      if (!selected.length) {
        setBacktestStatus("Choose at least one portfolio stock in Stock Research.", true);
        $("runButton").disabled = false;
        return;
      }
      if (unloaded.length) {
        setBacktestStatus(`Load price data first for: ${unloaded.slice(0, 8).join(", ")}${unloaded.length > 8 ? "..." : ""}`, true);
        $("runButton").disabled = false;
        return;
      }
      const weights = {};
      selected.forEach((symbol) => weights[symbol] = (state.weights[symbol] || 0) / 100);
      const preprocessor = selectedPreprocessor();
      const strategy = selectedStrategy();
      const neutralizer = selectedNeutralizer();
      const payload = {
        symbols: selected,
        weights,
        portfolioMode: $("portfolioMode").value,
        preprocessorId: preprocessor?.id || "none",
        preprocessorName: preprocessor?.name || $("preprocessorName").value,
        preprocessingCode: $("preprocessorCode").value,
        signalStrategy: "custom",
        strategyId: strategy?.id || "custom",
        strategyName: strategy?.name || $("strategyName").value,
        code: $("code").value,
        neutralizationMode: selectedNeutralizationMode(),
        neutralizerId: neutralizer?.id || "none",
        neutralizationCode: $("neutralizerCode").value,
        stocks: state.stocks,
        start: $("start").value,
        end: $("end").value,
        splitDate: $("splitDate").value,
        rebalanceFrequency: $("rebalanceFrequency").value,
        transactionCostBps: Number($("transactionCostBps").value || 0),
        slippageBps: Number($("slippageBps").value || 0)
      };
      try {
        const response = await fetch("/api/backtest", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        const result = await response.json();
        if (!response.ok) {
          setBacktestStatus(result.error || "Backtest failed", true);
          return;
        }
        renderResult(result);
        setBacktestStatus("Backtest complete.");
      } finally {
        $("runButton").disabled = false;
      }
    }

    function renderResult(result) {
      // Result refresh: fills Backtesting Results, Performance, Selected Stocks, and updates Stock Research.
      state.currentResult = result;
      state.portfolioRange = null;
      $("periodStatus").innerHTML = "";
      $("summary").textContent = "Backtesting Results";

      renderPortfolioChart();
      renderMetricsTable(result.metrics);
      renderYearlyTable(result.yearly);
      renderStocksTable(result.stocks, result.finalWeights);
      renderStockDetail();
    }

    function renderPortfolioChart() {
      // Backtesting Results box: switches between PnL, prices, weights, drawdown, Sharpe, and turnover plots.
      const result = state.currentResult;
      if (!result) return;
      const halfBookSize = result.capital?.halfBookSize || 10000000;
      const pnl = result.equity.map((point) => ({ date: point.date, value: point.value - halfBookSize }));
      const drawdown = computeDrawdown(result.equity, halfBookSize);
      const sharpe = expandingSharpeSeries(result.equity);
      const turnover = result.equity.map((point) => ({ date: point.date, value: point.turnover }));
      const type = $("portfolioChartType").value;
      // Backtesting Results chart numbers: height controls SVG viewBox height; font sizes control axes and Train/Test labels.
      const portfolioChartOptions = { height: 500, axisFontSize: 15, splitFontSize: 13 };
      const portfolioDates = result.equity.map((point) => point.date);
      const bounds = portfolioRangeBounds(portfolioDates);
      const ranged = (values) => filterByDateRange(values, bounds);
      if (type === "pnl") {
        drawChart("portfolioChart", "", [{ name: "PnL", values: ranged(pnl) }], result.periods.splitDate, "money", portfolioChartOptions);
      } else if (type === "prices") {
        drawChart("portfolioChart", "", Object.entries(result.prices).map(([name, values]) => ({ name, values: ranged(values) })), result.periods.splitDate, "percent", portfolioChartOptions);
      } else if (type === "weights") {
        drawChart("portfolioChart", "", Object.entries(result.weights).map(([name, values]) => ({ name, values: ranged(values) })), result.periods.splitDate, "percent", portfolioChartOptions);
      } else if (type === "drawdown") {
        drawChart("portfolioChart", "", [{ name: "Drawdown", values: ranged(drawdown) }], result.periods.splitDate, "percent", portfolioChartOptions);
      } else if (type === "sharpe") {
        drawChart("portfolioChart", "", [{ name: "Sharpe", values: ranged(sharpe) }], result.periods.splitDate, "number", portfolioChartOptions);
      } else if (type === "turnover") {
        drawChart("portfolioChart", "", [{ name: "Turnover", values: ranged(turnover) }], result.periods.splitDate, "percent", portfolioChartOptions);
      } else {
        drawChart("portfolioChart", "", [{ name: "PnL", values: ranged(pnl) }], result.periods.splitDate, "money", portfolioChartOptions);
      }
      appendPortfolioRangeControl("portfolioChart", portfolioDates);
    }

    function renderMetricsTable(metrics) {
      // Performance box: aggregate metric tiles for Aggregate, Training Period, and Testing Period.
      const group = (title, data) => `
        <div class="metric-group">
          <h3>${title}</h3>
          <div class="metric-grid">
            <div class="fact"><span>Returns</span><strong>${pct(data.annualized_return)}</strong></div>
            <div class="fact"><span>Sharpe</span><strong>${data.sharpe.toFixed(2)}</strong></div>
            <div class="fact"><span>IR</span><strong>${data.information_ratio.toFixed(2)}</strong></div>
            <div class="fact"><span>Turnover</span><strong>${pct(data.average_turnover)}</strong></div>
            <div class="fact"><span>Fitness</span><strong>${data.fitness.toFixed(2)}</strong></div>
            <div class="fact"><span>Drawdown</span><strong>${pct(data.max_drawdown)}</strong></div>
            <div class="fact"><span>Margin</span><strong>${bps(data.margin)}</strong></div>
          </div>
        </div>
      `;
      $("metricsTable").innerHTML = `
        <div class="metric-groups">
          ${group("Aggregate", metrics.full)}
          ${group("Training Period", metrics.training || metrics.is)}
          ${group("Testing Period", metrics.testing || metrics.os)}
        </div>
      `;
    }

    function renderYearlyTable(rows) {
      // Performance box: yearly metric table below the aggregate metric tiles.
      const body = (rows || []).map((row) => `
        <tr>
          <td>${row.year}</td>
          <td>${row.sharpe.toFixed(2)}</td>
          <td>${pct(row.turnover)}</td>
          <td>${row.fitness.toFixed(2)}</td>
          <td>${pct(row.returns)}</td>
          <td>${pct(row.drawdown)}</td>
          <td>${bps(row.margin)}</td>
          <td>${row.longCount}</td>
          <td>${row.shortCount}</td>
        </tr>
      `).join("");
      $("metricsTable").innerHTML += `
        <table>
          <tr><th>Year</th><th>Sharpe</th><th>Turnover</th><th>Fitness</th><th>Returns</th><th>Drawdown</th><th>Margin</th><th>Long Count</th><th>Short Count</th></tr>
          ${body}
        </table>
      `;
    }

    function renderStocksTable(stocks, finalWeights) {
      // Selected Stocks box: per-stock raw return and backtest position/activity table.
      const rows = stocks.map((stock) => `
        <tr>
          <td>${stock.symbol}</td>
          <td>${stock.region}</td>
          <td>${stock.sector}</td>
          <td>${pct(stock.totalReturn)}</td>
          <td>${pct(stock.avgAbsWeight)}</td>
          <td>${pct(stock.avgSignedWeight)}</td>
          <td>${pct(finalWeights[stock.symbol] || 0)}</td>
          <td>${stock.activeDays}</td>
          <td>${stock.longDays}</td>
          <td>${stock.shortDays}</td>
        </tr>
      `).join("");
      $("stocksTable").innerHTML = `<table><tr><th>Symbol</th><th>Region</th><th>Sector</th><th>Return</th><th>Avg Abs Weight</th><th>Avg Signed Weight</th><th>Final Weight</th><th>Active Days</th><th>Long Days</th><th>Short Days</th></tr>${rows}</table>`;
    }

    function renderStockDetail(options = {}) {
      // Stock Research box: renders the stock list, portfolio checkboxes/weights, and selected research figure.
      const previousScrollTop = options.preserveResearchScroll
        ? document.querySelector(".research-stock-list")?.scrollTop || 0
        : 0;
      const displayable = portfolioStocks();
      if (!state.compareSymbols.length && displayable.length && !state.researchSelectionCleared) {
        state.compareSymbols = [state.detailSymbol || displayable[0].symbol];
      }
      state.compareSymbols = state.compareSymbols.filter((symbol) => displayable.some((stock) => stock.symbol === symbol));
      if (!displayable.length) {
        $("stockResearchList").innerHTML = `<div class="meta">Select stocks from Available Stocks to inspect and weight them.</div>`;
        $("stockDetailChart").innerHTML = "";
        return;
      }
      const selectedStocks = state.compareSymbols.map((symbol) => state.stocks.find((item) => item.symbol === symbol)).filter(Boolean);
      if (!state.researchRange) {
        applyResearchWindowPreset(selectedStocks, false);
      }
      const rows = displayable.map((stock) => {
        const active = state.compareSymbols.includes(stock.symbol);
        const inPortfolio = state.selected.has(stock.symbol);
        return `
        <tr>
          <td><button class="icon-button ${active ? "active" : ""}" data-research-symbol="${stock.symbol}" title="Display ${stock.symbol}" aria-label="Display ${stock.symbol}">${active ? "-" : "+"}</button></td>
          <td class="truncate">${stock.symbol}</td>
	          <td class="truncate" title="${stock.name}">${stock.name}</td>
	          <td class="truncate">${stock.region}</td>
	          <td class="truncate">${stockIndustry(stock)}</td>
          <td><input type="checkbox" data-portfolio-symbol="${stock.symbol}" ${inPortfolio ? "checked" : ""} title="Use ${stock.symbol} in portfolio"></td>
	          <td><input type="number" data-research-weight="${stock.symbol}" value="${inPortfolio ? (state.weights[stock.symbol] ?? 0) : 0}" step="1" ${inPortfolio ? "" : "disabled"}></td>
	        </tr>
	      `;
      }).join("");
      const allCount = displayable.length;
      const portfolioCount = displayable.filter((stock) => state.selected.has(stock.symbol)).length;
      $("stockResearchList").innerHTML = `
        <div class="research-list-head">
          <h3>${state.compareSymbols.length || 0} Displayed</h3>
          <div class="research-actions">
            <button id="displayAllResearchStocks" type="button">Select All (${allCount})</button>
            <button id="displayPortfolioStocks" type="button">Select Portfolio (${portfolioCount})</button>
            <button id="clearResearchStocks" type="button">Clear</button>
          </div>
        </div>
        <div class="research-stock-list">
          <table>
            <colgroup>
              <col class="research-action-col">
              <col class="research-symbol-col">
	              <col class="research-name-col">
	              <col class="research-region-col">
	              <col class="research-industry-col">
              <col class="research-portfolio-col">
	              <col class="research-weight-col">
	            </colgroup>
	            <tr><th></th><th>Symbol</th><th>Name</th><th>Region</th><th>Industry</th><th>Portfolio</th><th>Weight %</th></tr>
	            ${rows}
	          </table>
        </div>
      `;
      if (options.preserveResearchScroll) {
        const list = document.querySelector(".research-stock-list");
        if (list) list.scrollTop = previousScrollTop;
      }
      document.querySelectorAll("[data-research-symbol]").forEach((node) => {
        node.addEventListener("click", () => {
          const symbol = node.dataset.researchSymbol;
          if (state.compareSymbols.includes(symbol)) {
            state.compareSymbols = state.compareSymbols.filter((item) => item !== symbol);
          } else {
            state.compareSymbols = [...state.compareSymbols, symbol];
          }
          state.researchSelectionCleared = !state.compareSymbols.length;
          renderStockDetail({ preserveResearchScroll: true });
        });
      });
      document.querySelectorAll("[data-research-weight]").forEach((node) => {
        node.addEventListener("input", () => {
          state.weights[node.dataset.researchWeight] = Number(node.value || 0);
        });
      });
      document.querySelectorAll("[data-portfolio-symbol]").forEach((node) => {
        node.addEventListener("change", () => {
          const symbol = node.dataset.portfolioSymbol;
          if (node.checked) {
            state.selected.add(symbol);
            state.weights[symbol] = state.weights[symbol] || 0;
          } else {
            state.selected.delete(symbol);
            state.weights[symbol] = 0;
          }
          renderStocks();
          renderStockDetail({ preserveResearchScroll: true });
        });
      });
      $("displayAllResearchStocks").addEventListener("click", () => {
        state.compareSymbols = displayable.map((stock) => stock.symbol);
        state.researchSelectionCleared = !state.compareSymbols.length;
        renderStockDetail({ preserveResearchScroll: true });
      });
      $("displayPortfolioStocks").addEventListener("click", () => {
        state.compareSymbols = displayable.filter((stock) => state.selected.has(stock.symbol)).map((stock) => stock.symbol);
        state.researchSelectionCleared = !state.compareSymbols.length;
        renderStockDetail({ preserveResearchScroll: true });
      });
      $("clearResearchStocks").addEventListener("click", () => {
        state.compareSymbols = [];
        state.researchSelectionCleared = true;
        renderStockDetail({ preserveResearchScroll: true });
      });

      const chartType = $("researchChartType").value;
      const bounds = researchRangeBounds(selectedStocks);
      if (chartType === "raw") {
        const series = selectedStocks.map((stock) => ({ name: stock.symbol, values: filterByDateRange(rawPriceSeries(stock.symbol), bounds) }));
        if (series.some((item) => item.values.length)) {
          // Stock Research price chart height. CSS .chart-frame > svg still caps the displayed SVG height.
          drawChart("stockDetailChart", "", series, null, "price", { showSplit: false, height: 560 });
          appendResearchRangeControl("stockDetailChart", selectedStocks);
        } else {
          $("stockDetailChart").innerHTML = `<div class="meta">No raw price data is available for this selection.</div>`;
        }
      } else {
        drawCandlestickCharts("stockDetailChart", selectedStocks, chartType.replace("candle_", ""), bounds);
      }
    }

    function rawPriceSeries(symbol) {
      const bars = rawBarsSeries(symbol);
      if (bars.length) {
        return bars.map((bar) => ({ date: bar.date, value: bar.close }));
      }
      const stock = state.stocks.find((item) => item.symbol === symbol);
      return [
        { date: $("start").value || "2019-01-02", value: stock?.firstClose || 0 },
        { date: $("end").value || $("start").value || "2019-01-02", value: stock?.lastClose || 0 }
      ];
    }

    function rawBarsSeries(symbol) {
      return state.rawBars?.[symbol] || [];
    }

    function currentResearchStocks() {
      return state.compareSymbols.map((symbol) => state.stocks.find((item) => item.symbol === symbol)).filter(Boolean);
    }

    function researchDomain(stocks) {
      const dates = stocks.flatMap((stock) => rawBarsSeries(stock.symbol).map((bar) => bar.date));
      return Array.from(new Set(dates)).sort();
    }

    function applyResearchWindowPreset(stocks, markCustom) {
      const value = $("researchWindow").value;
      if (value === "custom" && markCustom !== false) return;
      const dates = researchDomain(stocks);
      if (value === "all" || !dates.length) {
        state.researchRange = { start: 0, end: 100 };
        return;
      }
      const last = new Date(`${dates[dates.length - 1]}T00:00:00Z`).getTime();
      const firstVisible = last - Number(value) * 24 * 60 * 60 * 1000;
      const startIndex = Math.max(0, dates.findIndex((date) => new Date(`${date}T00:00:00Z`).getTime() >= firstVisible));
      state.researchRange = {
        start: dates.length > 1 ? Math.round(startIndex / (dates.length - 1) * 100) : 0,
        end: 100
      };
    }

    function researchRangeBounds(stocks) {
      const dates = researchDomain(stocks);
      if (!dates.length) return { startDate: null, endDate: null, dates };
      const range = state.researchRange || { start: 0, end: 100 };
      const startIndex = Math.max(0, Math.min(dates.length - 1, Math.floor(range.start / 100 * (dates.length - 1))));
      const endIndex = Math.max(startIndex, Math.min(dates.length - 1, Math.ceil(range.end / 100 * (dates.length - 1))));
      return {
        startDate: dates[startIndex],
        endDate: dates[endIndex],
        dates
      };
    }

    function filterByDateRange(points, bounds) {
      if (!bounds.startDate || !bounds.endDate) return points;
      return points.filter((point) => point.date >= bounds.startDate && point.date <= bounds.endDate);
    }

    function setResearchRange(start, end) {
      let nextStart = Math.max(0, Math.min(99, Number(start)));
      let nextEnd = Math.max(1, Math.min(100, Number(end)));
      if (nextStart >= nextEnd) {
        if (nextStart >= 99) {
          nextStart = 99;
          nextEnd = 100;
        } else {
          nextEnd = nextStart + 1;
        }
      }
      state.researchRange = { start: nextStart, end: nextEnd };
      $("researchWindow").value = "custom";
    }

    function setResearchRangeFromDates(startDate, endDate, dates) {
      if (!dates.length) return;
      const startIndex = Math.max(0, dates.findIndex((date) => date >= startDate));
      const resolvedStart = startIndex === -1 ? dates.length - 1 : startIndex;
      const endIndex = Math.max(0, dates.findIndex((date) => date >= endDate));
      const resolvedEnd = endIndex === -1 ? dates.length - 1 : endIndex;
      setResearchRange(
        Math.round(resolvedStart / Math.max(1, dates.length - 1) * 100),
        Math.round(resolvedEnd / Math.max(1, dates.length - 1) * 100)
      );
    }

    function appendResearchRangeControl(targetId, stocks) {
      // Stock Research figure: appends the draggable date-range bar and manual start/end date inputs.
      const bounds = researchRangeBounds(stocks);
      if (!bounds.dates.length) return;
      const range = state.researchRange || { start: 0, end: 100 };
      $(targetId).insertAdjacentHTML("beforeend", `
        <div class="range-zoom">
          <div class="range-head">
            <div class="range-date-inputs">
              <label>Start <input id="researchRangeStartDate" type="date" value="${bounds.startDate}" min="${bounds.dates[0]}" max="${bounds.dates[bounds.dates.length - 1]}"></label>
              <label>End <input id="researchRangeEndDate" type="date" value="${bounds.endDate}" min="${bounds.dates[0]}" max="${bounds.dates[bounds.dates.length - 1]}"></label>
            </div>
            <button class="range-reset" id="researchRangeReset" type="button">Reset</button>
          </div>
          <div class="range-track">
            <div class="range-track-line"></div>
            <div class="range-track-fill" id="researchRangeFill" style="left:${range.start}%; right:${100 - range.end}%;"></div>
            <input id="researchRangeStart" type="range" min="0" max="100" step="1" value="${range.start}" aria-label="Start date">
            <input id="researchRangeEnd" type="range" min="0" max="100" step="1" value="${range.end}" aria-label="End date">
          </div>
        </div>
      `);
      const refreshRangeControl = () => {
        const latest = researchRangeBounds(stocks);
        $("researchRangeFill").style.left = `${state.researchRange.start}%`;
        $("researchRangeFill").style.right = `${100 - state.researchRange.end}%`;
        $("researchRangeStart").value = state.researchRange.start;
        $("researchRangeEnd").value = state.researchRange.end;
        $("researchRangeStartDate").value = latest.startDate;
        $("researchRangeEndDate").value = latest.endDate;
      };
      $("researchRangeStart").addEventListener("input", (event) => {
        setResearchRange(event.target.value, state.researchRange.end);
        refreshRangeControl();
      });
      $("researchRangeEnd").addEventListener("input", (event) => {
        setResearchRange(state.researchRange.start, event.target.value);
        refreshRangeControl();
      });
      $("researchRangeStart").addEventListener("change", () => {
        renderStockDetail();
      });
      $("researchRangeEnd").addEventListener("change", () => {
        renderStockDetail();
      });
      $("researchRangeStartDate").addEventListener("change", (event) => {
        setResearchRangeFromDates(event.target.value, $("researchRangeEndDate").value, bounds.dates);
        renderStockDetail();
      });
      $("researchRangeEndDate").addEventListener("change", (event) => {
        setResearchRangeFromDates($("researchRangeStartDate").value, event.target.value, bounds.dates);
        renderStockDetail();
      });
      $("researchRangeReset").addEventListener("click", () => {
        $("researchWindow").value = "all";
        applyResearchWindowPreset(stocks, false);
        renderStockDetail();
      });
    }

    function portfolioRangeBounds(dates) {
      if (!dates.length) return { startDate: null, endDate: null, dates };
      const range = state.portfolioRange || { start: 0, end: 100 };
      const startIndex = Math.max(0, Math.min(dates.length - 1, Math.floor(range.start / 100 * (dates.length - 1))));
      const endIndex = Math.max(startIndex, Math.min(dates.length - 1, Math.ceil(range.end / 100 * (dates.length - 1))));
      return {
        startDate: dates[startIndex],
        endDate: dates[endIndex],
        dates
      };
    }

    function setPortfolioRange(start, end) {
      let nextStart = Math.max(0, Math.min(99, Number(start)));
      let nextEnd = Math.max(1, Math.min(100, Number(end)));
      if (nextStart >= nextEnd) {
        if (nextStart >= 99) {
          nextStart = 99;
          nextEnd = 100;
        } else {
          nextEnd = nextStart + 1;
        }
      }
      state.portfolioRange = { start: nextStart, end: nextEnd };
    }

    function setPortfolioRangeFromDates(startDate, endDate, dates) {
      if (!dates.length) return;
      const startIndex = Math.max(0, dates.findIndex((date) => date >= startDate));
      const resolvedStart = startIndex === -1 ? dates.length - 1 : startIndex;
      const endIndex = Math.max(0, dates.findIndex((date) => date >= endDate));
      const resolvedEnd = endIndex === -1 ? dates.length - 1 : endIndex;
      setPortfolioRange(
        Math.round(resolvedStart / Math.max(1, dates.length - 1) * 100),
        Math.round(resolvedEnd / Math.max(1, dates.length - 1) * 100)
      );
    }

    function appendPortfolioRangeControl(targetId, dates) {
      // Backtesting Results figure: appends the draggable date-range bar and manual start/end date inputs.
      if (!dates.length) return;
      const bounds = portfolioRangeBounds(dates);
      const range = state.portfolioRange || { start: 0, end: 100 };
      $(targetId).insertAdjacentHTML("beforeend", `
        <div class="range-zoom">
          <div class="range-head">
            <div class="range-date-inputs">
              <label>Start <input id="portfolioRangeStartDate" type="date" value="${bounds.startDate}" min="${dates[0]}" max="${dates[dates.length - 1]}"></label>
              <label>End <input id="portfolioRangeEndDate" type="date" value="${bounds.endDate}" min="${dates[0]}" max="${dates[dates.length - 1]}"></label>
            </div>
            <button class="range-reset" id="portfolioRangeReset" type="button">Reset</button>
          </div>
          <div class="range-track">
            <div class="range-track-line"></div>
            <div class="range-track-fill" id="portfolioRangeFill" style="left:${range.start}%; right:${100 - range.end}%;"></div>
            <input id="portfolioRangeStart" type="range" min="0" max="100" step="1" value="${range.start}" aria-label="Backtest start date">
            <input id="portfolioRangeEnd" type="range" min="0" max="100" step="1" value="${range.end}" aria-label="Backtest end date">
          </div>
        </div>
      `);
      const refreshRangeControl = () => {
        const latest = portfolioRangeBounds(dates);
        $("portfolioRangeFill").style.left = `${state.portfolioRange.start}%`;
        $("portfolioRangeFill").style.right = `${100 - state.portfolioRange.end}%`;
        $("portfolioRangeStart").value = state.portfolioRange.start;
        $("portfolioRangeEnd").value = state.portfolioRange.end;
        $("portfolioRangeStartDate").value = latest.startDate;
        $("portfolioRangeEndDate").value = latest.endDate;
      };
      $("portfolioRangeStart").addEventListener("input", (event) => {
        setPortfolioRange(event.target.value, state.portfolioRange?.end ?? 100);
        refreshRangeControl();
      });
      $("portfolioRangeEnd").addEventListener("input", (event) => {
        setPortfolioRange(state.portfolioRange?.start ?? 0, event.target.value);
        refreshRangeControl();
      });
      $("portfolioRangeStart").addEventListener("change", renderPortfolioChart);
      $("portfolioRangeEnd").addEventListener("change", renderPortfolioChart);
      $("portfolioRangeStartDate").addEventListener("change", (event) => {
        setPortfolioRangeFromDates(event.target.value, $("portfolioRangeEndDate").value, dates);
        renderPortfolioChart();
      });
      $("portfolioRangeEndDate").addEventListener("change", (event) => {
        setPortfolioRangeFromDates($("portfolioRangeStartDate").value, event.target.value, dates);
        renderPortfolioChart();
      });
      $("portfolioRangeReset").addEventListener("click", () => {
        state.portfolioRange = { start: 0, end: 100 };
        renderPortfolioChart();
      });
    }

    function drawCandlestickCharts(targetId, stocks, period, bounds) {
      // Stock Research figure: draws one combined candlestick chart for all selected stocks.
      const selected = stocks;
      if (!selected.length) {
        $(targetId).innerHTML = `<div class="meta">Select at least one stock to display.</div>`;
        return;
      }
      const series = selected
        .map((stock) => ({ name: stock.symbol, bars: aggregateBars(filterByDateRange(rawBarsSeries(stock.symbol), bounds), period) }))
        .filter((item) => item.bars.length);
      if (!series.length) {
        $(targetId).innerHTML = `<div class="meta">No candlestick data is available for this selection.</div>`;
        return;
      }
      $(targetId).innerHTML = candlestickSvg(series, period);
      appendResearchRangeControl(targetId, selected);
    }

    function aggregateBars(bars, period) {
      if (period === "day") return bars;
      const groups = new Map();
      bars.forEach((bar) => {
        const key = periodKey(bar.date, period);
        if (!groups.has(key)) {
          groups.set(key, { date: bar.date, open: bar.open, high: bar.high, low: bar.low, close: bar.close });
        } else {
          const current = groups.get(key);
          current.date = bar.date;
          current.high = Math.max(current.high, bar.high);
          current.low = Math.min(current.low, bar.low);
          current.close = bar.close;
        }
      });
      return Array.from(groups.values());
    }

    function periodKey(dateText, period) {
      const date = new Date(`${dateText}T00:00:00Z`);
      const year = date.getUTCFullYear();
      if (period === "year") return `${year}`;
      if (period === "quarter") return `${year}-Q${Math.floor(date.getUTCMonth() / 3) + 1}`;
      const day = date.getUTCDay() || 7;
      date.setUTCDate(date.getUTCDate() - day + 1);
      return date.toISOString().slice(0, 10);
    }

    function periodLabel(period) {
      if (period === "day") return "Daily";
      if (period === "week") return "Weekly";
      if (period === "quarter") return "Quarterly";
      return "Yearly";
    }

    function candlestickSvg(series, period) {
      // Candlestick chart numbers: width/height set the SVG viewBox; margins control plot padding.
      const width = 1080, height = 560, left = 64, right = 22, top = 34, bottom = 58;
      const prepared = series.map((item) => {
        const shown = item.bars;
        return {
          name: item.name,
          bars: shown
        };
      }).filter((item) => item.bars.length);
      const shownBars = prepared.flatMap((item) => item.bars);
      if (!shownBars.length) return `<div class="meta">No candlestick data</div>`;
      const dates = Array.from(new Set(shownBars.map((bar) => bar.date))).sort();
      const min = Math.min(...shownBars.map((bar) => bar.low));
      const max = Math.max(...shownBars.map((bar) => bar.high));
      const pad = Math.max((max - min) * 0.08, 0.01);
      const yMin = min - pad;
      const yMax = max + pad;
      const plotW = width - left - right;
      const plotH = height - top - bottom;
      const dateIndex = new Map(dates.map((date, index) => [date, index]));
      const x = (date) => left + (dateIndex.get(date) / Math.max(1, dates.length - 1)) * plotW;
      const y = (value) => top + ((yMax - value) / (yMax - yMin)) * plotH;
      const slotW = plotW / Math.max(1, dates.length);
      const candleW = Math.max(2, Math.min(9, slotW / Math.max(1, prepared.length) * 0.7));
      const candles = prepared.map((item, seriesIndex) => {
        const color = colors[seriesIndex % colors.length];
        const offset = prepared.length > 1 ? (seriesIndex - (prepared.length - 1) / 2) * candleW : 0;
        return item.bars.map((bar) => {
          const cx = x(bar.date) + offset;
          const up = bar.close >= bar.open;
          const stroke = color;
          const candleOpacity = up ? "0.28" : "0.9";
          const bodyTop = Math.min(y(bar.open), y(bar.close));
          const bodyHeight = Math.max(1, Math.abs(y(bar.open) - y(bar.close)));
          return `
            <g opacity="${candleOpacity}">
              <line x1="${cx.toFixed(1)}" y1="${y(bar.high).toFixed(1)}" x2="${cx.toFixed(1)}" y2="${y(bar.low).toFixed(1)}" stroke="${stroke}" stroke-width="1.2"/>
              <rect x="${(cx - candleW / 2).toFixed(1)}" y="${bodyTop.toFixed(1)}" width="${candleW.toFixed(1)}" height="${bodyHeight.toFixed(1)}" fill="${stroke}" stroke="${stroke}" stroke-width="${up ? "1.2" : "1.6"}"/>
            </g>
          `;
        }).join("");
      }).join("");
      const ticks = [yMin + pad, (yMin + yMax) / 2, yMax - pad].map((tick) => `
        <line x1="${left}" y1="${y(tick).toFixed(1)}" x2="${width - right}" y2="${y(tick).toFixed(1)}" stroke="#e8ebf0"/>
        <text x="${left - 8}" y="${(y(tick) + 4).toFixed(1)}" text-anchor="end" font-size="11" fill="#5d6877">${tick.toFixed(tick >= 100 ? 0 : 2)}</text>
      `).join("");
      const stockLegend = prepared.map((item, index) => {
        const color = colors[index % colors.length];
        return `<span class="legend-item"><span class="legend-swatch" style="background:${color}"></span>${item.name}</span>`;
      }).join("");
      const directionLegend = `<g opacity="0.28"><rect x="${width - 204}" y="12" width="12" height="9" fill="#17202a" stroke="#17202a"/></g><text x="${width - 186}" y="21" font-size="11" fill="#5d6877">Up</text><g opacity="0.9"><rect x="${width - 138}" y="12" width="12" height="9" fill="#17202a" stroke="#17202a" stroke-width="1.6"/></g><text x="${width - 120}" y="21" font-size="11" fill="#5d6877">Down</text>`;
      return `
        <div class="chart-frame">
          <svg viewBox="0 0 ${width} ${height}" role="img">
            <rect x="0" y="0" width="${width}" height="${height}" fill="#ffffff"/>
            ${directionLegend}
            ${ticks}
            <line x1="${left}" y1="${height - bottom}" x2="${width - right}" y2="${height - bottom}" stroke="#cfd5df"/>
            <line x1="${left}" y1="${top}" x2="${left}" y2="${height - bottom}" stroke="#cfd5df"/>
            ${candles}
            <text x="${left}" y="${height - 32}" font-size="11" fill="#5d6877">${dates[0]}</text>
            <text x="${width - right}" y="${height - 32}" text-anchor="end" font-size="11" fill="#5d6877">${dates[dates.length - 1]}</text>
          </svg>
        </div>
        <div class="chart-legend-viewport"><div class="chart-legend-strip">${stockLegend}</div></div>
      `;
    }

    function computeDrawdown(equity, halfBookSize) {
      let peakPnl = 0;
      return equity.map((point) => {
        const pnl = point.value - halfBookSize;
        peakPnl = Math.max(peakPnl, pnl);
        return { date: point.date, value: halfBookSize ? (pnl - peakPnl) / halfBookSize : 0 };
      });
    }

    function expandingSharpeSeries(points) {
      return points.map((point, index) => {
        const returns = points.slice(0, index + 1).map((item) => Number(item.dailyReturn) || 0);
        if (returns.length < 2) return { date: point.date, value: 0 };
        const mean = returns.reduce((sum, value) => sum + value, 0) / returns.length;
        const variance = returns.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (returns.length - 1);
        const volatility = Math.sqrt(variance);
        return { date: point.date, value: volatility ? (mean / volatility) * Math.sqrt(252) : 0 };
      });
    }

    function drawChart(targetId, title, series, splitDate, kind, options = {}) {
      // Generic line chart used by Stock Research and Backtesting Results.
      // width/height set SVG viewBox size; left/right/top/bottom are plot margins.
      const width = 1080, height = options.height || 400, left = 64, right = 22, top = 34, bottom = 44;
      const axisFontSize = options.axisFontSize || 15;
      const splitFontSize = options.splitFontSize || 11;
      const allDates = series.flatMap((item) => item.values.map((point) => point.date));
      const dates = Array.from(new Set(allDates)).sort();
      const values = series.flatMap((item) => item.values.map((point) => point.value));
      if (!dates.length || !values.length) {
        $(targetId).innerHTML = `${title ? `<h2>${title}</h2>` : ""}<div class="meta">No chart data</div>`;
        return;
      }
      let min = Math.min(...values), max = Math.max(...values);
      if (min === max) { min -= 1; max += 1; }
      const pad = (max - min) * 0.08;
      min -= pad; max += pad;
      const plotW = width - left - right, plotH = height - top - bottom;
      const dateIndex = new Map(dates.map((day, index) => [day, index]));
      const x = (day) => left + (dateIndex.get(day) / Math.max(1, dates.length - 1)) * plotW;
      const y = (value) => top + ((max - value) / (max - min)) * plotH;
      const fmt = (value) => {
        if (kind === "percent") return `${(value * 100).toFixed(0)}%`;
        if (kind === "price") return `${value.toFixed(value >= 100 ? 0 : 2)}`;
        if (kind === "number") return `${value.toFixed(2)}`;
        return `${(value / 1000).toFixed(0)}k`;
      };
      const ticks = [min, (min + max) / 2, max].map((tick) => `
        <line x1="${left}" y1="${y(tick)}" x2="${width - right}" y2="${y(tick)}" stroke="#e8ebf0"/>
        <text x="${left - 8}" y="${y(tick) + 4}" text-anchor="end" font-size="${axisFontSize}" fill="#5d6877">${fmt(tick)}</text>
      `).join("");
      const showSplit = options.showSplit !== false && splitDate;
      const splitIndex = showSplit ? Math.max(0, dates.filter((day) => day <= splitDate).length - 1) : 0;
      const splitX = left + (splitIndex / Math.max(1, dates.length - 1)) * plotW;
      const split = showSplit ? `
        <rect x="${left}" y="${top}" width="${Math.max(0, splitX - left)}" height="${plotH}" fill="#1b4d89" opacity="0.045"/>
        <rect x="${splitX}" y="${top}" width="${Math.max(0, width - right - splitX)}" height="${plotH}" fill="#00876c" opacity="0.055"/>
        <line x1="${splitX}" y1="${top}" x2="${splitX}" y2="${height - bottom}" stroke="#17202a" stroke-width="1.6" stroke-dasharray="4 4"/>
        <text x="${left + 8}" y="${top + 16}" font-size="${splitFontSize}" fill="#1b4d89">Train</text>
        <text x="${Math.min(splitX + 8, width - right - 48)}" y="${top + 16}" font-size="${splitFontSize}" fill="#00876c">Test</text>
      ` : "";
      const paths = series.map((item, index) => {
        const d = item.values.map((point, i) => `${i === 0 ? "M" : "L"} ${x(point.date).toFixed(1)} ${y(point.value).toFixed(1)}`).join(" ");
        return `<path d="${d}" fill="none" stroke="${colors[index % colors.length]}" stroke-width="2"/>`;
      }).join("");
      const legend = series.map((item, index) => {
        const color = colors[index % colors.length];
        return `<span class="legend-item"><span class="legend-swatch" style="background:${color}"></span>${item.name}</span>`;
      }).join("");
      $(targetId).innerHTML = `
        ${title ? `<h2>${title}</h2>` : ""}
        <div class="chart-frame">
          <svg viewBox="0 0 ${width} ${height}" role="img">
            <rect x="0" y="0" width="${width}" height="${height}" fill="#ffffff"/>
            ${split}
            ${ticks}
            <line x1="${left}" y1="${height - bottom}" x2="${width - right}" y2="${height - bottom}" stroke="#cfd5df"/>
            <line x1="${left}" y1="${top}" x2="${left}" y2="${height - bottom}" stroke="#cfd5df"/>
            <text x="${left}" y="${height - 18}" font-size="${axisFontSize}" fill="#5d6877">${dates[0]}</text>
            <text x="${width - right}" y="${height - 18}" text-anchor="end" font-size="${axisFontSize}" fill="#5d6877">${dates[dates.length - 1]}</text>
            ${paths}
          </svg>
        </div>
        <div class="chart-legend-viewport"><div class="chart-legend-strip">${legend}</div></div>`;
    }

    init();
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
