from sklearn.utils.estimator_checks import parametrize_with_checks

from rangebranch import AdaptiveRangeTreeClassifier, AdaptiveRangeTreeRegressor


@parametrize_with_checks(
    [
        AdaptiveRangeTreeClassifier(
            max_depth=2,
            min_samples_leaf=2,
            min_samples_split=4,
        ),
        AdaptiveRangeTreeRegressor(
            max_depth=2,
            min_samples_leaf=2,
            min_samples_split=4,
        ),
    ]
)
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)
