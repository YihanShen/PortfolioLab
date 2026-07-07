from __future__ import annotations

import math
from dataclasses import dataclass

from portfoliolab.backtest import BacktestResult


@dataclass(frozen=True)
class PerformanceMetrics:
    total_return: float
    annualized_return: float
    total_pnl: float
    annualized_pnl: float
    annualized_volatility: float
    information_ratio: float
    sharpe: float
    sortino: float
    max_drawdown: float
    average_turnover: float
    max_gross_exposure: float
    margin: float
    fitness: float
    total_dollars_traded: float

    def as_dict(self) -> dict[str, float]:
        return {
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "total_pnl": self.total_pnl,
            "annualized_pnl": self.annualized_pnl,
            "annualized_volatility": self.annualized_volatility,
            "information_ratio": self.information_ratio,
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "max_drawdown": self.max_drawdown,
            "average_turnover": self.average_turnover,
            "max_gross_exposure": self.max_gross_exposure,
            "margin": self.margin,
            "fitness": self.fitness,
            "total_dollars_traded": self.total_dollars_traded,
        }


def evaluate(result: BacktestResult) -> PerformanceMetrics:
    values = result.values
    returns = result.returns
    if not values:
        raise ValueError("Cannot evaluate an empty backtest result")

    half_book_size = result.config.initial_cash
    daily_pnls = [point.daily_pnl for point in result.equity_curve]
    total_pnl = sum(daily_pnls)
    total_return = _ratio(total_pnl, half_book_size)
    average_daily_return = sum(returns) / len(returns)
    annualized_pnl = (sum(daily_pnls) / len(daily_pnls)) * 252.0
    annualized_return = _ratio(annualized_pnl, half_book_size)
    daily_volatility = _stddev(returns)
    annualized_volatility = daily_volatility * math.sqrt(252.0)
    information_ratio = _ratio(average_daily_return, daily_volatility)
    sharpe = information_ratio * math.sqrt(252.0)

    downside = [min(0.0, item) for item in returns]
    sortino = _ratio(average_daily_return, _stddev(downside)) * math.sqrt(252.0)
    max_drawdown = _pnl_drawdown(daily_pnls, half_book_size)
    average_turnover = sum(point.turnover for point in result.equity_curve) / len(result.equity_curve)
    max_gross_exposure = max(point.gross_exposure for point in result.equity_curve)
    total_dollars_traded = sum(point.dollars_traded for point in result.equity_curve)
    margin = _ratio(total_pnl, total_dollars_traded)
    fitness = sharpe * math.sqrt(abs(annualized_return) / max(average_turnover, 0.125))

    return PerformanceMetrics(
        total_return=total_return,
        annualized_return=annualized_return,
        total_pnl=total_pnl,
        annualized_pnl=annualized_pnl,
        annualized_volatility=annualized_volatility,
        information_ratio=information_ratio,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_drawdown,
        average_turnover=average_turnover,
        max_gross_exposure=max_gross_exposure,
        margin=margin,
        fitness=fitness,
        total_dollars_traded=total_dollars_traded,
    )


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
