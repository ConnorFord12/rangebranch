"""Public inspection helpers for fitted RangeBranch estimators."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def export_text(estimator: Any, decimals: int = 3, show_stats: bool = True) -> str:
    """Return the estimator as readable multi-range decision rules."""
    return estimator.export_text(decimals=decimals, show_stats=show_stats)


def export_rules(estimator: Any, decimals: int = 6) -> list[dict[str, Any]]:
    """Return one JSON-compatible rule dictionary per terminal leaf."""
    return estimator.get_rules(decimals=decimals)


def explain_prediction(estimator: Any, x: Sequence[float]) -> dict[str, Any]:
    """Explain the exact root-to-leaf route taken by one observation."""
    return estimator.explain_prediction(x)
