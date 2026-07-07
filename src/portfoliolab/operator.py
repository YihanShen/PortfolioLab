from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from types import ModuleType
from typing import Protocol

from portfoliolab.data import MarketData


@dataclass(frozen=True)
class OperatorContext:
    rebalance_frequency: str
    max_gross_exposure: float


class WeightOperator(Protocol):
    def compute(self, data: MarketData, as_of: date, context: OperatorContext) -> dict[str, float]:
        """Return target weights to hold after the close on as_of."""


def load_operator(path: str | Path) -> WeightOperator:
    module = _load_module(Path(path))
    strategy_type = getattr(module, "Strategy", None)
    if strategy_type is None:
        raise ValueError(f"{path} must expose a Strategy class")
    strategy = strategy_type()
    if not hasattr(strategy, "compute"):
        raise ValueError(f"{path} Strategy must define compute(data, as_of, context)")
    return strategy


def _load_module(path: Path) -> ModuleType:
    if not path.exists():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load operator module from {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

