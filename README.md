# RangeBranch

[![Tests](https://github.com/ConnorFord12/rangebranch/actions/workflows/tests.yml/badge.svg)](https://github.com/ConnorFord12/rangebranch/actions/workflows/tests.yml)
[![PyPI version](https://img.shields.io/pypi/v/rangebranch.svg)](https://pypi.org/project/rangebranch/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)](https://github.com/ConnorFord12/rangebranch/releases)

RangeBranch is an experimental scikit-learn-compatible decision-tree library
for learning **adaptive multi-range splits** on numerical features.

Instead of representing a numerical decision as a chain of binary questions:

```text
weight < 10?
weight < 40?
weight < 75?
```

one RangeBranch node can express the operating regimes directly:

```text
product_weight
├── < 10
├── [10, 40)
├── [40, 75)
└── >= 75
```

The goal is not to replace high-performance ensembles. It is to provide a
compact, inspectable estimator for tabular problems where numerical ranges have
operational meaning.

> **Status:** `0.1.0a1` is an alpha release. The local range partition is
> optimized exactly on a candidate grid, while the overall tree is still grown
> greedily. Statistical stability regularization is planned but is not claimed
> by this release.

## Features

- Classification and regression
- Two-to-K-way splits selected separately at each node
- Dynamic programming over contiguous candidate intervals
- Scikit-learn `fit`, `predict`, `predict_proba`, `score`, cloning, and pipelines
- NumPy arrays and pandas DataFrames
- Missing-value routing
- Sample weights
- Normalized complexity penalty across differently scaled targets
- Smoothed leaf probabilities
- Text trees, JSON-compatible leaf rules, decision paths, and explanations
- Matplotlib visualization with interval-labeled branches

## Installation

From a built release:

```bash
pip install rangebranch
```

For local development:

```bash
python -m venv .venv
.venv/Scripts/activate  # Windows
python -m pip install --upgrade pip
python -m pip install -e ".[dev,examples]"
```

On macOS/Linux, activate with `source .venv/bin/activate`.

## Classification

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from rangebranch import AdaptiveRangeTreeClassifier, plot_tree

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    stratify=y,
    random_state=42,
)

model = AdaptiveRangeTreeClassifier(
    max_depth=4,
    max_branches=4,
    max_bins=40,
    min_samples_leaf=30,
    branch_penalty=0.01,
    probability_smoothing=1.0,
    random_state=42,
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

print("Balanced accuracy:", balanced_accuracy_score(y_test, predictions))
print("ROC-AUC:", roc_auc_score(y_test, probabilities))
print(model.export_text())

plot_tree(model, save_path="range_tree.png")
```

## Regression

```python
from rangebranch import AdaptiveRangeTreeRegressor

model = AdaptiveRangeTreeRegressor(
    max_depth=4,
    max_branches=4,
    min_samples_leaf=30,
    branch_penalty=0.01,
    random_state=42,
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

Because `normalize_gain=True` by default, `branch_penalty` operates on the
fraction of node impurity improved by a candidate split. This makes the penalty
more comparable between targets measured in different units.

## Inspection

```python
from rangebranch import explain_prediction, export_rules, export_text

print(export_text(model))

rules = export_rules(model)
explanation = explain_prediction(model, X_test.iloc[0])
```

A classification rule contains its complete root-to-leaf conditions, support,
prediction, impurity, and class probabilities.

## Product-spoilage example

The example creates synthetic product data with weight, dimensions, category
IDs, quantity, cost, COGS, shelf life, warehouse dwell time, temperature,
humidity, distance, and handling events. It trains both classification and
regression trees and exports all results.

```bash
python examples/product_spoilage.py \
    --samples 6000 \
    --output-dir sart_product_results
```

Outputs include:

```text
synthetic_product_data.csv
scored_test_data.csv
metrics.json
classification_rules.json
regression_rules.json
classification_tree.txt
regression_tree.txt
classification_tree.png
regression_tree.png
```

## Principal parameters

| Parameter | Meaning |
|---|---|
| `max_depth` | Maximum root-to-leaf depth |
| `max_branches` | Maximum numerical children per node |
| `max_bins` | Maximum atomic intervals evaluated per feature |
| `min_samples_split` | Minimum effective sample weight required to split |
| `min_samples_leaf` | Minimum effective sample weight per numerical branch |
| `min_impurity_decrease` | Minimum unnormalized impurity improvement |
| `branch_penalty` | Penalty for every additional boundary |
| `normalize_gain` | Normalize gain by the parent impurity before penalization |
| `probability_smoothing` | Added pseudo-count per class in each leaf |
| `missing_strategy` | Route missing values to a separate or largest child |
| `max_features` | Number or fraction of candidate features at each node |

## Algorithm summary

For every candidate feature at a node, RangeBranch:

1. Sorts its observed values and creates at most `max_bins` atomic intervals.
2. Calculates sufficient target statistics for those intervals.
3. Uses dynamic programming to find the lowest-loss contiguous partition for
   each branch count from 2 through `max_branches`.
4. Penalizes additional boundaries.
5. Selects the best feature and range partition.
6. Recursively repeats the process for each child.

For thresholds \(t_1 < \dots < t_{K-1}\), the node creates:

```text
(-∞, t1), [t1, t2), ..., [tK-1, ∞)
```

Training is approximately `O(p * K * m²)` at a node, where `p` is the number
of candidate features, `K` is `max_branches`, and `m` is the number of atomic
intervals.

## Appropriate uses

RangeBranch is most appropriate when:

- The data is tabular and low-to-medium dimensional.
- Several regimes of one numerical measurement are meaningful.
- Rules will be reviewed or implemented by people.
- A compact single tree is more important than maximum ensemble accuracy.
- Non-monotonic effects are expected.

Potential applications include product spoilage, process-control limits,
predictive maintenance, insurance exposure bands, fraud amount regimes,
promotion tiers, agriculture, energy demand, and interpretable risk triage.

## Limitations

- All features are currently treated as numeric and ordered.
- Unordered category IDs should not be passed without intentional encoding.
- The estimator is predictive, not causal; a learned threshold does not prove
  that changing a feature will change the outcome.
- Deep or highly branched trees can become visually and statistically complex.
- This alpha does not guarantee cross-version model-pickle compatibility.
- The pure Python/NumPy splitter is not designed for hundreds of millions of rows.

## Development

```bash
python -m pytest
python -m pytest --cov=rangebranch --cov-report=term-missing
python -m ruff check .
python -m build
python -m twine check dist/*
```

See [docs/releasing.md](docs/releasing.md) for TestPyPI and production-release
instructions, and [docs/research-roadmap.md](docs/research-roadmap.md) for the
planned statistical-stability work.

## License

MIT
