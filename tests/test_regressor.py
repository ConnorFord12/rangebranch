import numpy as np
from sklearn.metrics import r2_score

from rangebranch import AdaptiveRangeTreeRegressor


def test_regression_learns_piecewise_ranges():
    rng = np.random.default_rng(21)
    x = rng.uniform(0, 100, 1_500)
    X = np.c_[x, rng.normal(size=len(x))]
    y = np.select([x < 25, x < 60], [2.0, 11.0], default=5.0)
    y += rng.normal(0, 0.15, len(x))

    model = AdaptiveRangeTreeRegressor(
        max_depth=1,
        max_branches=3,
        # Search every observed boundary so this test isolates the tree's
        # multi-range behavior rather than the optional quantile approximation.
        max_bins=2_048,
        min_samples_leaf=40,
        branch_penalty=0.001,
    ).fit(X, y)

    assert len(model.tree_.thresholds) == 2
    assert np.allclose(model.tree_.thresholds, [25, 60], atol=2.5)
    assert r2_score(y, model.predict(X)) > 0.98


def test_normalized_penalty_is_target_scale_invariant():
    rng = np.random.default_rng(9)
    x = rng.uniform(0, 20, 500)
    X = x.reshape(-1, 1)
    y = np.where(x < 5, 2.0, np.where(x < 13, 7.0, 4.0))

    params = dict(
        max_depth=1,
        max_branches=3,
        min_samples_leaf=20,
        branch_penalty=0.01,
        normalize_gain=True,
    )
    small = AdaptiveRangeTreeRegressor(**params).fit(X, y)
    large = AdaptiveRangeTreeRegressor(**params).fit(X, y * 10_000)

    assert np.allclose(small.tree_.thresholds, large.tree_.thresholds)
    assert np.array_equal(small.apply(X), large.apply(X))
