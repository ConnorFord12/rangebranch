import numpy as np
import pytest

from rangebranch import AdaptiveRangeTreeClassifier


def test_all_zero_sample_weights_are_rejected():
    with pytest.raises(ValueError, match="weight.*zero"):
        AdaptiveRangeTreeClassifier().fit(
            [[0.0], [1.0]],
            [0, 1],
            sample_weight=[0.0, 0.0],
        )


def test_missing_values_can_receive_separate_child():
    X = np.r_[
        np.arange(40, dtype=float),
        np.arange(60, 100, dtype=float),
        np.full(30, np.nan),
    ].reshape(-1, 1)
    y = np.r_[np.zeros(40, dtype=int), np.ones(40, dtype=int), np.ones(30, dtype=int)]

    model = AdaptiveRangeTreeClassifier(
        max_depth=1,
        max_branches=2,
        min_samples_leaf=20,
        missing_strategy="separate",
        branch_penalty=0.0,
    ).fit(X, y)

    assert model.tree_.missing_child == 2
    assert len(model.tree_.children) == 3
    assert model.apply([[np.nan]])[0] == model.tree_.children[2].node_id


def test_integer_sample_weights_match_repeated_rows():
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    weights = np.array([1, 2, 3, 1, 2, 3], dtype=float)
    repeated_index = np.repeat(np.arange(len(X)), weights.astype(int))

    params = dict(
        max_depth=2,
        max_branches=3,
        min_samples_leaf=1,
        branch_penalty=0.0,
        probability_smoothing=0.0,
    )
    weighted = AdaptiveRangeTreeClassifier(**params).fit(X, y, sample_weight=weights)
    repeated = AdaptiveRangeTreeClassifier(**params).fit(X[repeated_index], y[repeated_index])

    assert np.array_equal(weighted.predict(X), repeated.predict(X))
    assert np.allclose(weighted.predict_proba(X), repeated.predict_proba(X))
