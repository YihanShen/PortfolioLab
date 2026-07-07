from __future__ import annotations

import csv
import io
from datetime import date
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from portfoliolab.data import Bar, MarketData


class DataProviderError(RuntimeError):
    """Raised when a market data provider cannot return usable data."""


NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
NASDAQ_OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"


def fetch_nasdaq_us_symbol_catalog() -> list[dict[str, str]]:
    """Fetch lightweight U.S. symbol metadata without price history."""

    try:
        nasdaq_rows = _read_symbol_directory(NASDAQ_LISTED_URL)
        other_rows = _read_symbol_directory(NASDAQ_OTHER_LISTED_URL)
    except (OSError, URLError) as exc:
        raise DataProviderError(f"Could not refresh Nasdaq symbol catalog: {exc}") from exc

    catalog: list[dict[str, str]] = []
    for row in nasdaq_rows:
        symbol = _clean_symbol(row.get("Symbol", ""))
        name = row.get("Security Name", symbol).strip()
        if not _is_common_equity(symbol, name, row.get("ETF", "")):
            continue
        catalog.append(
            {
                "symbol": symbol,
                "provider_symbol": _yahoo_symbol(symbol),
                "name": name,
                "region": "United States",
                "source": "Yahoo Finance",
                "sector": "",
                "industry": "",
                "exchange": "NASDAQ",
            }
        )

    exchange_names = {
        "A": "NYSE American",
        "N": "NYSE",
        "P": "NYSE Arca",
        "Z": "BATS",
        "V": "IEX",
    }
    for row in other_rows:
        symbol = _clean_symbol(row.get("ACT Symbol", ""))
        name = row.get("Security Name", symbol).strip()
        if not _is_common_equity(symbol, name, row.get("ETF", "")):
            continue
        sector = ""
        exchange = exchange_names.get(row.get("Exchange", ""), row.get("Exchange", ""))
        catalog.append(
            {
                "symbol": symbol,
                "provider_symbol": _yahoo_symbol(symbol),
                "name": name,
                "region": "United States",
                "source": "Yahoo Finance",
                "sector": sector,
                "industry": "",
                "exchange": exchange,
            }
        )

    unique = {item["symbol"]: item for item in catalog}
    return sorted(unique.values(), key=lambda item: (item["symbol"], item["name"]))


def fetch_yahoo_data(
    symbol_map: dict[str, str],
    *,
    start: date | None = None,
    period: str = "10y",
) -> MarketData:
    """Fetch daily OHLCV data from Yahoo Finance via the optional yfinance package."""

    try:
        import yfinance as yf
    except ImportError as exc:
        raise DataProviderError("Install yfinance to refresh Yahoo Finance data: pip install yfinance") from exc

    if not symbol_map:
        raise DataProviderError("No symbols were provided for the Yahoo Finance refresh")

    yahoo_symbols = list(dict.fromkeys(symbol_map.values()))
    inverse = {provider_symbol: internal for internal, provider_symbol in symbol_map.items()}
    kwargs = {
        "tickers": yahoo_symbols,
        "interval": "1d",
        "group_by": "ticker",
        "auto_adjust": False,
        "progress": False,
        "threads": True,
    }
    if start is not None:
        kwargs["start"] = start.isoformat()
    else:
        kwargs["period"] = period

    frame = yf.download(**kwargs)
    if frame is None or frame.empty:
        raise DataProviderError("Yahoo Finance returned no data")

    bars: list[Bar] = []
    for provider_symbol in yahoo_symbols:
        symbol_frame = _symbol_frame(frame, provider_symbol, len(yahoo_symbols) == 1)
        if symbol_frame is None or symbol_frame.empty:
            continue
        internal_symbol = inverse[provider_symbol]
        for index, row in symbol_frame.iterrows():
            values = {name.lower(): _cell(row, name) for name in ("Open", "High", "Low", "Close", "Volume")}
            if any(values[name] is None for name in ("open", "high", "low", "close")):
                continue
            bars.append(
                Bar(
                    date=index.date(),
                    symbol=internal_symbol,
                    open=float(values["open"]),
                    high=float(values["high"]),
                    low=float(values["low"]),
                    close=float(values["close"]),
                    volume=float(values["volume"] or 0.0),
                )
            )

    if not bars:
        raise DataProviderError("Yahoo Finance returned no usable OHLCV rows")
    return MarketData(bars)


def lookup_yahoo_symbols(symbols: list[str]) -> list[dict[str, str]]:
    """Validate ticker availability on Yahoo Finance without loading full history."""

    try:
        import yfinance as yf
    except ImportError as exc:
        raise DataProviderError("Install yfinance to search Yahoo Finance symbols: pip install yfinance") from exc

    unique_symbols = list(dict.fromkeys(_clean_symbol(symbol) for symbol in symbols if _clean_symbol(symbol)))
    if not unique_symbols:
        raise DataProviderError("Enter at least one ticker symbol to search")

    found: list[dict[str, str]] = []
    for raw_symbol in unique_symbols:
        provider_symbol = _manual_yahoo_symbol(raw_symbol)
        try:
            ticker = yf.Ticker(provider_symbol)
            history = ticker.history(period="5d", interval="1d")
        except Exception:
            continue
        if history is None or history.empty:
            continue

        info: dict[str, object] = {}
        try:
            info = ticker.get_info() or {}
        except Exception:
            info = {}

        name = str(info.get("shortName") or info.get("longName") or raw_symbol).strip()
        exchange = str(info.get("exchange") or info.get("fullExchangeName") or "").strip()
        sector = str(info.get("sector") or "").strip()
        industry = str(info.get("industry") or "").strip()
        country = str(info.get("country") or "").strip()
        found.append(
            {
                "symbol": provider_symbol,
                "provider_symbol": provider_symbol,
                "name": name,
                "region": country or "Unknown",
                "source": "Yahoo Finance",
                "sector": sector,
                "industry": industry,
                "exchange": exchange,
            }
        )

    if not found:
        raise DataProviderError("Yahoo Finance did not return any available symbols for that search")
    return found


def write_market_data_csv(data: MarketData, path: str | Path) -> Path:
    """Persist normalized market data so the app can restart from real-data cache."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["date", "symbol", "open", "high", "low", "close", "volume"])
        writer.writeheader()
        for day in data.dates:
            for symbol in data.symbols:
                bar = data.bar(day, symbol)
                if bar is None:
                    continue
                writer.writerow(
                    {
                        "date": bar.date.isoformat(),
                        "symbol": bar.symbol,
                        "open": bar.open,
                        "high": bar.high,
                        "low": bar.low,
                        "close": bar.close,
                        "volume": bar.volume,
                    }
                )
    return target


def _symbol_frame(frame, provider_symbol: str, single_symbol: bool):
    if single_symbol:
        return frame
    if provider_symbol in frame.columns.get_level_values(0):
        return frame[provider_symbol]
    return None


def _read_symbol_directory(url: str) -> list[dict[str, str]]:
    with urlopen(url, timeout=20) as response:
        text = response.read().decode("utf-8", errors="replace")
    rows = []
    for row in csv.DictReader(io.StringIO(text), delimiter="|"):
        first_value = next(iter(row.values()), "")
        if first_value and first_value.startswith("File Creation Time"):
            continue
        rows.append(row)
    return rows


def _clean_symbol(symbol: str) -> str:
    return symbol.strip().upper()


def _yahoo_symbol(symbol: str) -> str:
    return symbol.replace(".", "-").replace("/", "-").strip().upper()


def _manual_yahoo_symbol(symbol: str) -> str:
    cleaned = _clean_symbol(symbol).replace("/", "-")
    parts = cleaned.split(".")
    if len(parts) == 2 and len(parts[1]) == 1:
        return f"{parts[0]}-{parts[1]}"
    return cleaned


def _is_common_equity(symbol: str, name: str, etf_flag: str = "") -> bool:
    if not symbol or etf_flag == "Y":
        return False
    if "$" in symbol or "^" in symbol:
        return False

    lower_name = name.casefold()
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


def _cell(row, column: str):
    try:
        value = row[column]
    except KeyError:
        return None
    if value != value:
        return None
    return value
