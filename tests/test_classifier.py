import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score

from rangebranch import AdaptiveRangeTreeClassifier


def test_recovers_four_known_numeric_ranges():
    rng = np.random.default_rng(123)
    x = rng.uniform(0, 100, 2_000)
    X = np.c_[x, rng.normal(size=len(x))]
    y = np.select([x < 20, x < 45, x < 75], [0, 1, 2], default=0)

    model = AdaptiveRangeTreeClassifier(
        max_depth=1,
        max_branches=4,
        max_bins=64,
        min_samples_leaf=50,
        branch_penalty=0.0001,
        probability_smoothing=0.0,
        random_state=7,
    ).fit(X, y)

    assert len(model.tree_.thresholds) == 3
    assert np.allclose(model.tree_.thresholds, [20, 45, 75], atol=2.5)
    assert accuracy_score(y, model.predict(X)) > 0.98
    assert clone(model).get_params()["max_branches"] == 4


def test_dataframe_names_multiclass_and_probabilities():
    X = pd.DataFrame(
        {
            "weight": [1, 2, 3, 10, 11, 12, 50, 51, 52] * 10,
            "cost": [2, 2, 3, 5, 5, 6, 8, 8, 9] * 10,
        }
    )
    y = np.array(["small"] * 3 + ["medium"] * 3 + ["large"] * 3)
    y = np.tile(y, 10)

    model = AdaptiveRangeTreeClassifier(
        max_depth=2,
        max_branches=3,
        min_samples_leaf=5,
        branch_penalty=0.0,
    ).fit(X, y)

    probabilities = model.predict_proba(X)
    assert model.tree_.feature_name in X.columns
    assert probabilities.shape == (len(X), 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert set(model.predict(X)) == {"small", "medium", "large"}


def test_probability_smoothing_avoids_exact_zeroes():
    X = np.arange(100, dtype=float).reshape(-1, 1)
    y = np.r_[np.zeros(50, dtype=int), np.ones(50, dtype=int)]
    model = AdaptiveRangeTreeClassifier(
        max_depth=1,
        min_samples_leaf=10,
        probability_smoothing=1.0,
        branch_penalty=0.0,
    ).fit(X, y)

    probabilities = model.predict_proba([[1.0], [99.0]])
    assert np.all(probabilities > 0)
    assert np.all(probabilities < 1)
