# Research roadmap

RangeBranch's basic multiway numerical split is not itself new. The research
direction is to produce **stable, uncertainty-aware, backward-traversable range
rules** and validate their usefulness against established tree methods.

## Proposed stability estimator

For each candidate node:

1. Generate bootstrap or subsample replicates.
2. Record the selected feature and ordered thresholds.
3. Cluster nearby threshold estimates.
4. Estimate feature-selection frequency and boundary uncertainty.
5. Penalize partitions with low recurrence or high boundary variance.
6. Report confidence intervals and support for every retained boundary.

The resulting estimator should be introduced under the separate public name:

```python
StableAdaptiveRangeTreeClassifier
StableAdaptiveRangeTreeRegressor
```

The current aliases exist only for MVP compatibility and must not be described
as statistically stabilized.

## Primary hypothesis

At comparable predictive performance, stability-regularized adaptive range
trees produce more reproducible and concise operational rules than binary CART
and unregularized multiway trees.

## Baselines

- Pruned CART
- Globally discretized CART
- CHAID or a reproducible equivalent
- Explainable Boosting Machines
- An unregularized Adaptive Range Tree
- Optimal sparse trees when dataset size permits

## Metrics

### Prediction

- Balanced accuracy, ROC-AUC, PR-AUC, log loss
- RMSE, MAE, and R-squared
- Calibration and Brier score

### Complexity

- Nodes and leaves
- Mean path length
- Mean boundary comparisons
- Distinct features per path

### Stability

- Feature-selection agreement
- Boundary distance and confidence-interval width
- Leaf-assignment agreement
- Rule overlap and prediction agreement

### Rule utility

- Coverage, precision, lift, and support
- Estimated action cost
- Expected operational or economic value

## Experimental sequence

1. Synthetic piecewise targets with known boundaries.
2. Noise, imbalance, correlated features, missingness, and drift experiments.
3. Repeated nested cross-validation on public tabular datasets.
4. Ablation studies for branching, normalization, smoothing, and stability.
5. A domain case study with permission-safe or synthetic spoilage data.

No claim of universal superiority should be made. The intended result is a
well-characterized accuracy–stability–interpretability tradeoff.
