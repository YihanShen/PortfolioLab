from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date

from portfoliolab.backtest import BacktestConfig, BacktestEngine, BacktestResult
from portfoliolab.data import MarketData
from portfoliolab.metrics import PerformanceMetrics, evaluate
from portfoliolab.operator import WeightOperator


@dataclass(frozen=True)
class SplitEvaluation:
    split_date: date
    full: BacktestResult
    in_sample: BacktestResult
    out_of_sample: BacktestResult
    in_sample_metrics: PerformanceMetrics
    out_of_sample_metrics: PerformanceMetrics


def run_is_os_evaluation(
    data: MarketData,
    operator: WeightOperator,
    config: BacktestConfig,
    split_date: date,
) -> SplitEvaluation:
    dates = data.window_dates(config.start, config.end)
    if not dates:
        raise ValueError("No dates available for the requested evaluation window")

    os_start = next((day for day in dates if day > split_date), None)
    if os_start is None:
        raise ValueError("split_date must leave at least two out-of-sample trading dates")

    is_config = replace(config, end=split_date)
    os_config = replace(config, start=os_start)
    engine = BacktestEngine()
    full = engine.run(data, operator, config)
    in_sample = engine.run(data, operator, is_config)
    out_of_sample = engine.run(data, operator, os_config)

    return SplitEvaluation(
        split_date=split_date,
        full=full,
        in_sample=in_sample,
        out_of_sample=out_of_sample,
        in_sample_metrics=evaluate(in_sample),
        out_of_sample_metrics=evaluate(out_of_sample),
    )
