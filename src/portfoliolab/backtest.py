from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from portfoliolab.data import MarketData
from portfoliolab.operator import OperatorContext, WeightOperator


@dataclass(frozen=True)
class BacktestConfig:
    start: date | None = None
    end: date | None = None
    initial_cash: float = 10_000_000.0
    rebalance_frequency: str = "monthly"
    transaction_cost_bps: float = 5.0
    slippage_bps: float = 2.0
    max_gross_exposure: float = 2.0


@dataclass(frozen=True)
class EquityPoint:
    date: date
    value: float
    daily_pnl: float
    daily_return: float
    turnover: float
    dollars_traded: float
    gross_exposure: float
    weights: dict[str, float]


@dataclass(frozen=True)
class BacktestResult:
    equity_curve: list[EquityPoint]
    final_weights: dict[str, float]
    config: BacktestConfig

    @property
    def returns(self) -> list[float]:
        return [point.daily_return for point in self.equity_curve]

    @property
    def values(self) -> list[float]:
        return [point.value for point in self.equity_curve]


class BacktestEngine:
    def run(
        self,
        data: MarketData,
        operator: WeightOperator,
        config: BacktestConfig,
    ) -> BacktestResult:
        dates = data.window_dates(config.start, config.end)
        if len(dates) < 2:
            raise ValueError("Backtest requires at least two trading dates")

        context = OperatorContext(
            rebalance_frequency=config.rebalance_frequency,
            max_gross_exposure=config.max_gross_exposure,
        )
        half_book_size = config.initial_cash
        book_size = half_book_size * 2.0
        cumulative_pnl = 0.0
        weights: dict[str, float] = {}
        equity_curve: list[EquityPoint] = []
        last_rebalance_key: tuple[int, ...] | None = None
        cost_rate = (config.transaction_cost_bps + config.slippage_bps) / 10_000.0

        for index, today in enumerate(dates[:-1]):
            tomorrow = dates[index + 1]
            dollars_traded = 0.0
            daily_turnover = 0.0
            trading_cost = 0.0

            rebalance_key = _period_key(today, config.rebalance_frequency)
            if last_rebalance_key != rebalance_key:
                raw_target = operator.compute(data, today, context)
                target = _clean_weights(raw_target, data, today, config.max_gross_exposure)
                weight_delta = _turnover(weights, target)
                dollars_traded = half_book_size * weight_delta
                daily_turnover = dollars_traded / book_size if book_size else 0.0
                trading_cost = dollars_traded * cost_rate
                weights = target
                last_rebalance_key = rebalance_key

            holding_return = _portfolio_return(data, today, tomorrow, weights)
            daily_pnl = (half_book_size * holding_return) - trading_cost
            cumulative_pnl += daily_pnl
            value = half_book_size + cumulative_pnl
            daily_return = daily_pnl / half_book_size if half_book_size else 0.0
            equity_curve.append(
                EquityPoint(
                    date=tomorrow,
                    value=value,
                    daily_pnl=daily_pnl,
                    daily_return=daily_return,
                    turnover=daily_turnover,
                    dollars_traded=dollars_traded,
                    gross_exposure=sum(abs(weight) for weight in weights.values()),
                    weights=dict(weights),
                )
            )

        return BacktestResult(equity_curve=equity_curve, final_weights=weights, config=config)


def _period_key(day: date, frequency: str) -> tuple[int, ...]:
    if frequency == "daily":
        return (day.year, day.month, day.day)
    if frequency == "weekly":
        iso = day.isocalendar()
        return (iso.year, iso.week)
    if frequency == "monthly":
        return (day.year, day.month)
    raise ValueError("rebalance_frequency must be one of: daily, weekly, monthly")


def _clean_weights(
    weights: dict[str, float],
    data: MarketData,
    day: date,
    max_gross_exposure: float,
) -> dict[str, float]:
    cleaned = {
        symbol.upper(): float(weight)
        for symbol, weight in weights.items()
        if weight and data.close(day, symbol.upper()) is not None
    }
    gross = sum(abs(weight) for weight in cleaned.values())
    if gross <= max_gross_exposure or gross == 0:
        return cleaned
    scale = max_gross_exposure / gross
    return {symbol: weight * scale for symbol, weight in cleaned.items()}


def _turnover(previous: dict[str, float], target: dict[str, float]) -> float:
    symbols = set(previous).union(target)
    return sum(abs(target.get(symbol, 0.0) - previous.get(symbol, 0.0)) for symbol in symbols)


def _portfolio_return(data: MarketData, today: date, tomorrow: date, weights: dict[str, float]) -> float:
    total = 0.0
    for symbol, weight in weights.items():
        start_price = data.close(today, symbol)
        end_price = data.close(tomorrow, symbol)
        if start_price is None or end_price is None or start_price <= 0:
            continue
        total += weight * ((end_price / start_price) - 1.0)
    return total
