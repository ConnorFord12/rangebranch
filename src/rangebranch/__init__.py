"""RangeBranch: interpretable trees with adaptive multi-range splits."""

from ._tree import (
    AdaptiveRangeTreeClassifier,
    AdaptiveRangeTreeRegressor,
    StableAdaptiveRangeTreeClassifier,
    StableAdaptiveRangeTreeRegressor,
    plot_tree,
)
from .inspection import explain_prediction, export_rules, export_text

__version__ = "0.1.0a1"

__all__ = [
    "AdaptiveRangeTreeClassifier",
    "AdaptiveRangeTreeRegressor",
    "StableAdaptiveRangeTreeClassifier",
    "StableAdaptiveRangeTreeRegressor",
    "explain_prediction",
    "export_rules",
    "export_text",
    "plot_tree",
]
