from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from portfoliolab.data import Bar, MarketData


@dataclass(frozen=True)
class FixedWeightStrategy:
    weights: dict[str, float]

    def compute(self, data, as_of, context):
        return dict(self.weights)


@dataclass(frozen=True)
class NeutralizedStrategy:
    strategy: object
    neutralize: Callable[[dict[str, float], dict[str, str]], dict[str, float]]
    groups: dict[str, str]

    def compute(self, data, as_of, context):
        raw = self.strategy.compute(data, as_of, context)
        alpha = {symbol: float(raw.get(symbol, 0.0)) for symbol in self.groups}
        adjusted = self.neutralize(alpha, dict(self.groups))
        return _scale_to_gross(adjusted, context.max_gross_exposure)


@dataclass(frozen=True)
class SelectedMomentumStrategy:
    symbols: list[str]
    top_n: int = 5
    lookback_days: int = 252
    skip_recent_days: int = 21

    def compute(self, data, as_of, context):
        scores = []
        for symbol in self.symbols:
            score = data.return_over(
                symbol,
                as_of,
                lookback=self.lookback_days,
                skip_recent=self.skip_recent_days,
            )
            if score is not None and score > 0:
                scores.append((score, symbol))

        selected = [symbol for _, symbol in sorted(scores, reverse=True)[: self.top_n]]
        if not selected:
            return {}

        weight = min(context.max_gross_exposure, 1.0) / len(selected)
        return {symbol: weight for symbol in selected}


@dataclass(frozen=True)
class SelectedMeanReversionStrategy:
    symbols: list[str]
    bottom_n: int = 5
    lookback_days: int = 21

    def compute(self, data, as_of, context):
        scores = []
        for symbol in self.symbols:
            score = data.return_over(symbol, as_of, lookback=self.lookback_days)
            if score is not None:
                scores.append((score, symbol))

        selected_scores = sorted(scores)[: self.bottom_n]
        if not selected_scores:
            return {}

        strongest_selected_score = selected_scores[-1][0]
        raw_alpha = {
            symbol: max(strongest_selected_score - score, 0.0)
            for score, symbol in selected_scores
        }
        if sum(raw_alpha.values()) == 0:
            raw_alpha = {symbol: 1.0 for _, symbol in selected_scores}
        return _scale_to_gross(raw_alpha, min(context.max_gross_exposure, 1.0))


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    cleaned = {symbol.upper(): float(weight) for symbol, weight in weights.items() if float(weight) != 0}
    gross = sum(abs(weight) for weight in cleaned.values())
    if gross == 0:
        return {}
    return {symbol: weight / gross for symbol, weight in cleaned.items()}


def equal_weights(symbols: list[str]) -> dict[str, float]:
    cleaned = [symbol.upper() for symbol in symbols if symbol]
    if not cleaned:
        return {}
    weight = 1.0 / len(cleaned)
    return {symbol: weight for symbol in cleaned}


def _scale_to_gross(weights: dict[str, float], target_gross: float) -> dict[str, float]:
    cleaned = {symbol.upper(): float(weight) for symbol, weight in weights.items() if float(weight) != 0}
    gross = sum(abs(weight) for weight in cleaned.values())
    if gross == 0:
        return {}
    scale = target_gross / gross
    return {symbol: weight * scale for symbol, weight in cleaned.items()}


def _code_namespace(extra: dict[str, object] | None = None) -> dict[str, object]:
    try:
        import numpy as np
        import pandas as pd
    except ImportError:
        np = None
        pd = None

    namespace = {
        "Bar": Bar,
        "MarketData": MarketData,
        "np": np,
        "pd": pd,
    }
    if extra:
        namespace.update(extra)
    return namespace


def load_inline_strategy(code: str, selected_symbols: list[str], base_weights: dict[str, float]):
    namespace = _code_namespace(
        {
            "SELECTED_SYMBOLS": list(selected_symbols),
            "BASE_WEIGHTS": dict(base_weights),
        }
    )
    exec(code, namespace)
    strategy_type = namespace.get("Strategy")
    if strategy_type is None:
        raise ValueError("Custom code must define a Strategy class")
    strategy = strategy_type()
    if not hasattr(strategy, "compute"):
        raise ValueError("Strategy must define compute(data, as_of, context)")
    return strategy


def load_inline_preprocessor(code: str):
    namespace = _code_namespace()
    exec(code, namespace)
    preprocess = namespace.get("preprocess")
    if preprocess is None:
        raise ValueError("Preprocessing code must define preprocess(data, symbols, context)")
    if not callable(preprocess):
        raise ValueError("preprocess must be callable")
    return preprocess


def load_inline_neutralizer(code: str):
    namespace = _code_namespace()
    exec(code, namespace)
    neutralize = namespace.get("neutralize")
    if neutralize is None:
        raise ValueError("Neutralization code must define neutralize(alpha, groups)")
    if not callable(neutralize):
        raise ValueError("neutralize must be callable")
    group_for = namespace.get("group_for")
    if group_for is not None and not callable(group_for):
        raise ValueError("group_for must be callable")
    return neutralize, group_for
