import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from rangebranch import AdaptiveRangeTreeClassifier


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("max_branches", 1),
        ("max_bins", 1),
        ("min_samples_leaf", 0),
        ("branch_penalty", -0.1),
        ("probability_smoothing", -1.0),
        ("missing_strategy", "unknown"),
    ],
)
def test_invalid_parameters_raise(parameter, value):
    model = AdaptiveRangeTreeClassifier(**{parameter: value})
    with pytest.raises(ValueError):
        model.fit(np.arange(20).reshape(-1, 1), np.array([0, 1] * 10))


def test_predict_before_fit_raises():
    with pytest.raises(NotFittedError):
        AdaptiveRangeTreeClassifier().predict([[1.0]])


def test_non_numeric_features_are_rejected():
    with pytest.raises(ValueError, match="numeric"):
        AdaptiveRangeTreeClassifier().fit([["red"], ["blue"]], [0, 1])
