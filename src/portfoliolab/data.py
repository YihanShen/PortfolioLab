from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path


@dataclass(frozen=True)
class Bar:
    date: date
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class MarketData:
    """Daily OHLCV data normalized into a date/symbol lookup."""

    def __init__(self, bars: list[Bar]):
        self._bars_by_date: dict[date, dict[str, Bar]] = {}
        self._dates_by_symbol: dict[str, list[date]] = {}

        for bar in sorted(bars, key=lambda item: (item.date, item.symbol)):
            self._bars_by_date.setdefault(bar.date, {})[bar.symbol] = bar
            self._dates_by_symbol.setdefault(bar.symbol, []).append(bar.date)

        self._dates = sorted(self._bars_by_date)
        self._symbols = sorted(self._dates_by_symbol)

    @property
    def dates(self) -> list[date]:
        return list(self._dates)

    @property
    def symbols(self) -> list[str]:
        return list(self._symbols)

    def window_dates(self, start: date | None = None, end: date | None = None) -> list[date]:
        return [
            day
            for day in self._dates
            if (start is None or day >= start) and (end is None or day <= end)
        ]

    def bar(self, day: date, symbol: str) -> Bar | None:
        return self._bars_by_date.get(day, {}).get(symbol)

    def close(self, day: date, symbol: str) -> float | None:
        bar = self.bar(day, symbol)
        return None if bar is None else bar.close

    def history(self, symbol: str, as_of: date, lookback: int, skip_recent: int = 0) -> list[Bar]:
        dates = [day for day in self._dates_by_symbol.get(symbol, []) if day <= as_of]
        if skip_recent:
            dates = dates[:-skip_recent]
        if lookback:
            dates = dates[-lookback:]
        return [self._bars_by_date[day][symbol] for day in dates]

    def return_over(self, symbol: str, as_of: date, lookback: int, skip_recent: int = 0) -> float | None:
        bars = self.history(symbol, as_of, lookback=lookback, skip_recent=skip_recent)
        if len(bars) < 2:
            return None
        first = bars[0].close
        last = bars[-1].close
        if first <= 0:
            return None
        return (last / first) - 1.0

    def to_frame(self):
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("Install pandas to use DataFrame preprocessing: pip install pandas") from exc

        rows = []
        for day in self.dates:
            for symbol in self.symbols:
                bar = self.bar(day, symbol)
                if bar is None:
                    continue
                rows.append(
                    {
                        "date": bar.date,
                        "symbol": bar.symbol,
                        "open": bar.open,
                        "high": bar.high,
                        "low": bar.low,
                        "close": bar.close,
                        "volume": bar.volume,
                    }
                )
        return pd.DataFrame(rows, columns=["date", "symbol", "open", "high", "low", "close", "volume"])

    @classmethod
    def from_frame(cls, frame) -> MarketData:
        required = {"date", "symbol", "open", "high", "low", "close", "volume"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"DataFrame is missing required columns: {', '.join(sorted(missing))}")

        bars = []
        for row in frame.to_dict("records"):
            day = row["date"]
            if hasattr(day, "date"):
                day = day.date()
            elif isinstance(day, str):
                day = parse_date(day[:10])
            bars.append(
                Bar(
                    date=day,
                    symbol=str(row["symbol"]).strip().upper(),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"] or 0.0),
                )
            )
        return cls(bars)

    @classmethod
    def from_csv(cls, path: str | Path) -> MarketData:
        bars: list[Bar] = []
        with Path(path).open(newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"date", "symbol", "open", "high", "low", "close", "volume"}
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing))}")

            for row in reader:
                bars.append(
                    Bar(
                        date=parse_date(row["date"]),
                        symbol=row["symbol"].strip().upper(),
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        volume=float(row["volume"]),
                    )
                )
        return cls(bars)

    @classmethod
    def demo(cls) -> MarketData:
        """Create deterministic sample prices for a smoke-test backtest."""

        start = date(2018, 1, 2)
        symbols = {
            "AAPL": (100.0, 0.00042, 0.00075),
            "AMZN": (86.0, 0.00036, 0.00095),
            "ASML": (132.0, 0.00034, 0.00090),
            "BABA": (72.0, 0.00008, 0.00120),
            "JNJ": (91.0, 0.00018, 0.00045),
            "KO": (55.0, 0.00015, 0.00040),
            "LVMH": (120.0, 0.00026, 0.00075),
            "META": (80.0, 0.00031, 0.00105),
            "MSFT": (105.0, 0.00045, 0.00070),
            "NESN": (78.0, 0.00012, 0.00038),
            "NOVO": (66.0, 0.00039, 0.00065),
            "NVDA": (62.0, 0.00058, 0.00135),
            "PG": (84.0, 0.00017, 0.00042),
            "SAP": (96.0, 0.00023, 0.00068),
            "SONY": (70.0, 0.00020, 0.00085),
            "TM": (74.0, 0.00016, 0.00058),
            "TSM": (64.0, 0.00040, 0.00095),
            "WMT": (69.0, 0.00019, 0.00036),
        }
        bars: list[Bar] = []
        prices = {symbol: seed[0] for symbol, seed in symbols.items()}
        trading_day = 0

        for offset in range(365 * 6):
            day = start + timedelta(days=offset)
            if day.weekday() >= 5:
                continue

            for index, (symbol, (_, drift, vol)) in enumerate(symbols.items()):
                cycle = math.sin((trading_day + index * 11) / 59.0) * vol
                shock = math.cos((trading_day + index * 7) / 17.0) * vol * 0.25
                daily_return = drift + cycle + shock
                previous = prices[symbol]
                close = max(1.0, previous * (1.0 + daily_return))
                high = max(previous, close) * 1.003
                low = min(previous, close) * 0.997
                bars.append(
                    Bar(
                        date=day,
                        symbol=symbol,
                        open=previous,
                        high=high,
                        low=low,
                        close=close,
                        volume=1_000_000 + trading_day * 10 + index * 25_000,
                    )
                )
                prices[symbol] = close
            trading_day += 1

        return cls(bars)


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()
