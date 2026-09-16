# Changelog

All notable changes to RangeBranch will be documented here.

The project follows semantic versioning. Versions below `1.0.0` may introduce
documented API changes as the estimator is validated.

## 0.1.0a1 - 2026-09-16

### Added

- Adaptive two-to-K-way numerical splitting for classification and regression.
- Dynamic-programming optimization of contiguous ranges at each node.
- Scikit-learn-compatible estimator API and estimator tags.
- Missing-value routing and sample-weight support.
- Normalized split-complexity scoring.
- Configurable Dirichlet/Laplace smoothing for class probabilities.
- Text, JSON-rule, prediction-path, and graphical inspection interfaces.
- Product-spoilage example, tests, CI, and release workflows.

### Known limitations

- Features must be numeric; numeric category identifiers are treated as ordered.
- Tree growth is greedy between nodes, although each local range partition is optimized.
- The implementation is pure Python/NumPy and not yet optimized for very large data.
- Statistical bootstrap stability regularization is not included in this release.
