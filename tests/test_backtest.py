from __future__ import annotations

import math
from datetime import date
from unittest import TestCase

from portfoliolab.backtest import BacktestConfig, BacktestEngine, BacktestResult, EquityPoint
from portfoliolab.data import MarketData
from portfoliolab.evaluation import run_is_os_evaluation
from portfoliolab.metrics import evaluate


class EqualWeightStrategy:
    def compute(self, data, as_of, context):
        symbols = data.symbols[:2]
        return {symbol: 1 / len(symbols) for symbol in symbols}


class BacktestEngineTest(TestCase):
    def test_demo_backtest_produces_metrics(self):
        result = BacktestEngine().run(
            MarketData.demo(),
            EqualWeightStrategy(),
            BacktestConfig(start=date(2019, 1, 2), end=date(2020, 12, 31)),
        )

        metrics = evaluate(result)

        self.assertGreater(len(result.equity_curve), 200)
        self.assertNotEqual(metrics.total_return, 0)
        self.assertGreaterEqual(metrics.max_gross_exposure, 0)
        self.assertTrue(hasattr(result.equity_curve[0], "daily_pnl"))
        self.assertTrue(hasattr(result.equity_curve[0], "dollars_traded"))

    def test_split_evaluation_returns_is_and_os_metrics(self):
        evaluation = run_is_os_evaluation(
            MarketData.demo(),
            EqualWeightStrategy(),
            BacktestConfig(start=date(2019, 1, 2), end=date(2022, 12, 30)),
            split_date=date(2020, 12, 31),
        )

        self.assertGreater(len(evaluation.in_sample.equity_curve), 200)
        self.assertGreater(len(evaluation.out_of_sample.equity_curve), 200)

    def test_metrics_follow_pnl_book_size_definitions(self):
        config = BacktestConfig(initial_cash=10_000_000)
        result = BacktestResult(
            equity_curve=[
                EquityPoint(
                    date=date(2024, 1, 2),
                    value=10_010_000,
                    daily_pnl=10_000,
                    daily_return=0.001,
                    turnover=0.25,
                    dollars_traded=5_000_000,
                    gross_exposure=1.0,
                    weights={},
                ),
                EquityPoint(
                    date=date(2024, 1, 3),
                    value=10_005_000,
                    daily_pnl=-5_000,
                    daily_return=-0.0005,
                    turnover=0.0,
                    dollars_traded=0,
                    gross_exposure=1.0,
                    weights={},
                ),
            ],
            final_weights={},
            config=config,
        )

        metrics = evaluate(result)
        expected_annualized_return = (((10_000 - 5_000) / 2) * 252) / 10_000_000
        expected_ir = ((0.001 - 0.0005) / 2) / math.sqrt(((0.001 - 0.00025) ** 2 + (-0.0005 - 0.00025) ** 2))

        self.assertAlmostEqual(metrics.total_return, 0.0005)
        self.assertAlmostEqual(metrics.annualized_return, expected_annualized_return)
        self.assertAlmostEqual(metrics.information_ratio, expected_ir)
        self.assertAlmostEqual(metrics.sharpe, math.sqrt(252) * expected_ir)
        self.assertAlmostEqual(metrics.average_turnover, 0.125)
        self.assertAlmostEqual(metrics.margin, 0.001)
        self.assertAlmostEqual(metrics.max_drawdown, 0.0005)
        self.assertAlmostEqual(
            metrics.fitness,
            metrics.sharpe * math.sqrt(abs(metrics.annualized_return) / max(metrics.average_turnover, 0.125)),
        )
