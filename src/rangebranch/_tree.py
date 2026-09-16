"""Adaptive Range Tree estimators.

This module implements interpretable, axis-aligned decision trees whose
numeric decision nodes may create two or more ordered interval branches.

The implementation is intentionally self-contained and sklearn-compatible so
that the algorithm can be tested before turning it into a full package.

Dependencies
------------
numpy, scikit-learn, matplotlib

Example
-------
    from rangebranch import (
        AdaptiveRangeTreeClassifier,
        plot_tree,
    )

    model = AdaptiveRangeTreeClassifier(
        max_depth=3,
        max_branches=4,
        min_samples_leaf=20,
        branch_penalty=0.01,
        random_state=42,
    )
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)
    print(model.export_text())
    plot_tree(model, save_path="sart_tree.png")

Notes
-----
* All input features must currently be numeric. Missing values are supported.
* Each node greedily chooses one feature, but dynamic programming finds the
  best contiguous K-way partition for that feature on the candidate grid.
* Native categorical subsets, pruning, stability regularization, and formal
  counterfactual search are planned extensions.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import column_or_1d, validate_data

ArrayLike = np.ndarray | Sequence[Sequence[float]]


@dataclass
class _TreeNode:
    """Internal representation of one range-tree node."""

    node_id: int
    depth: int
    n_samples: int
    weighted_n_samples: float
    impurity: float
    prediction: int | float
    probabilities: np.ndarray | None = None
    feature_index: int | None = None
    feature_name: str | None = None
    thresholds: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    children: list[_TreeNode] = field(default_factory=list)
    missing_child: int | None = None
    gain: float = 0.0

    @property
    def is_leaf(self) -> bool:
        return self.feature_index is None or not self.children


@dataclass
class _Split:
    feature_index: int
    thresholds: np.ndarray
    assignments: np.ndarray
    missing_child: int
    raw_gain: float
    penalized_gain: float


class _BaseAdaptiveRangeTree(BaseEstimator):
    """Shared implementation for classifier and regressor variants."""

    _task: str = "base"

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags

    def __init__(
        self,
        *,
        max_depth: int | None = 4,
        max_branches: int = 4,
        max_bins: int = 32,
        min_samples_split: int = 20,
        min_samples_leaf: int = 10,
        min_impurity_decrease: float = 0.0,
        branch_penalty: float = 0.005,
        normalize_gain: bool = True,
        probability_smoothing: float = 1.0,
        criterion: str | None = None,
        missing_strategy: str = "separate",
        max_features: int | float | str | None = None,
        random_state: int | None = None,
    ) -> None:
        self.max_depth = max_depth
        self.max_branches = max_branches
        self.max_bins = max_bins
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.branch_penalty = branch_penalty
        self.normalize_gain = normalize_gain
        self.probability_smoothing = probability_smoothing
        self.criterion = criterion
        self.missing_strategy = missing_strategy
        self.max_features = max_features
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Public estimator API
    # ------------------------------------------------------------------
    def fit(
        self,
        X: ArrayLike,
        y: Sequence[Any],
        sample_weight: Sequence[float] | None = None,
    ) -> _BaseAdaptiveRangeTree:
        """Fit the adaptive multi-range tree.

        Parameters
        ----------
        X:
            Numeric matrix or pandas DataFrame of shape (n_samples, n_features).
        y:
            Classification labels or numeric regression targets.
        sample_weight:
            Optional non-negative observation weights.
        """
        self._validate_hyperparameters()
        X_array, feature_names = self._prepare_X(X, fitting=True)
        y_array = column_or_1d(y, warn=True)
        if X_array.shape[0] != y_array.shape[0]:
            raise ValueError("X and y contain different numbers of samples.")
        if X_array.shape[0] == 0:
            raise ValueError("At least one training sample is required.")

        weights = self._prepare_sample_weight(sample_weight, X_array.shape[0])
        positive_weight = weights > 0
        X_array = X_array[positive_weight]
        y_array = y_array[positive_weight]
        weights = weights[positive_weight]
        self.display_feature_names_ = np.asarray(feature_names, dtype=object)
        self._rng = np.random.default_rng(self.random_state)
        self._next_node_id = 0

        if self._task == "classification":
            check_classification_targets(y_array)
            self.classes_, encoded = np.unique(y_array, return_inverse=True)
            self.n_classes_ = len(self.classes_)
            self._y_training = encoded.astype(np.int64)
            self.criterion_ = self.criterion or "gini"
            if self.criterion_ not in {"gini", "entropy", "log_loss"}:
                raise ValueError("Classifier criterion must be 'gini', 'entropy', or 'log_loss'.")
        elif self._task == "regression":
            try:
                self._y_training = y_array.astype(float)
            except (TypeError, ValueError) as exc:
                raise ValueError("Regression targets must be numeric.") from exc
            if not np.all(np.isfinite(self._y_training)):
                raise ValueError("Regression targets cannot contain NaN or infinity.")
            self.criterion_ = self.criterion or "squared_error"
            if self.criterion_ not in {"squared_error", "variance"}:
                raise ValueError("Regressor criterion must be 'squared_error' or 'variance'.")
        else:
            raise RuntimeError("Unknown estimator task.")

        all_indices = np.arange(X_array.shape[0], dtype=np.int64)
        self.tree_ = self._grow_tree(X_array, self._y_training, weights, all_indices, depth=0)
        self.n_nodes_ = self._next_node_id
        self.n_leaves_ = sum(node.is_leaf for node in self._iter_nodes())
        self.max_depth_ = max(node.depth for node in self._iter_nodes())

        # Training references are no longer needed after construction.
        del self._y_training
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict class labels or regression values."""
        self._check_is_fitted()
        X_array, _ = self._prepare_X(X, fitting=False)
        encoded_or_values = np.asarray([self._leaf_for_row(row).prediction for row in X_array])
        if self._task == "classification":
            return self.classes_[encoded_or_values.astype(int)]
        return encoded_or_values.astype(float)

    def apply(self, X: ArrayLike) -> np.ndarray:
        """Return the terminal node id reached by every sample."""
        self._check_is_fitted()
        X_array, _ = self._prepare_X(X, fitting=False)
        return np.asarray([self._leaf_for_row(row).node_id for row in X_array], dtype=int)

    def decision_path(self, X: ArrayLike) -> list[list[int]]:
        """Return ordered node ids traversed by each observation."""
        self._check_is_fitted()
        X_array, _ = self._prepare_X(X, fitting=False)
        paths: list[list[int]] = []
        for row in X_array:
            _, nodes, _ = self._trace_row(row)
            paths.append([node.node_id for node in nodes])
        return paths

    def explain_prediction(self, x: Sequence[float]) -> dict[str, Any]:
        """Return the interval decisions and terminal statistics for one row."""
        self._check_is_fitted()
        row = np.asarray(x, dtype=float)
        if row.ndim != 1 or row.shape[0] != self.n_features_in_:
            raise ValueError(f"x must contain exactly {self.n_features_in_} feature values.")
        leaf, nodes, branch_indices = self._trace_row(row)
        decisions = []
        for node, branch_idx in zip(nodes[:-1], branch_indices, strict=True):
            decisions.append(
                {
                    "node_id": node.node_id,
                    "feature": node.feature_name,
                    "value": float(row[node.feature_index])
                    if not np.isnan(row[node.feature_index])
                    else np.nan,
                    "interval": self._branch_label(node, branch_idx),
                    "gain": node.gain,
                }
            )
        result: dict[str, Any] = {
            "prediction": self._display_prediction(leaf),
            "leaf_id": leaf.node_id,
            "leaf_samples": leaf.n_samples,
            "leaf_impurity": leaf.impurity,
            "decisions": decisions,
        }
        if self._task == "classification":
            result["probabilities"] = {
                label.item() if hasattr(label, "item") else label: float(prob)
                for label, prob in zip(
                    self.classes_, leaf.probabilities, strict=True
                )
            }
        return result

    def export_text(self, decimals: int = 3, show_stats: bool = True) -> str:
        """Return a readable range-based representation of the fitted tree."""
        self._check_is_fitted()
        lines: list[str] = []

        def walk(node: _TreeNode, prefix: str) -> None:
            if node.is_leaf:
                details = self._leaf_text(node, decimals)
                if show_stats:
                    details += f" | n={node.n_samples}, impurity={node.impurity:.{decimals}f}"
                lines.append(prefix + "Predict " + details)
                return
            node_line = f"{node.feature_name} (gain={node.gain:.{decimals}f}"
            if show_stats:
                node_line += f", n={node.n_samples}"
            node_line += ")"
            lines.append(prefix + node_line)
            for idx, child in enumerate(node.children):
                lines.append(prefix + f"|-- {self._branch_label(node, idx, decimals)}")
                walk(child, prefix + "|   ")

        walk(self.tree_, "")
        return "\n".join(lines)

    def get_rules(self, decimals: int = 6) -> list[dict[str, Any]]:
        """Return one backward-traversable rule dictionary per leaf."""
        self._check_is_fitted()
        rules: list[dict[str, Any]] = []

        def walk(node: _TreeNode, conditions: list[dict[str, Any]]) -> None:
            if node.is_leaf:
                rule: dict[str, Any] = {
                    "leaf_id": node.node_id,
                    "conditions": list(conditions),
                    "prediction": self._display_prediction(node),
                    "n_samples": node.n_samples,
                    "impurity": round(node.impurity, decimals),
                }
                if self._task == "classification":
                    rule["probabilities"] = {
                        str(label): round(float(prob), decimals)
                        for label, prob in zip(
                            self.classes_, node.probabilities, strict=True
                        )
                    }
                rules.append(rule)
                return
            for child_idx, child in enumerate(node.children):
                condition = {
                    "feature": node.feature_name,
                    "feature_index": int(node.feature_index),
                    "interval": self._branch_label(node, child_idx, decimals),
                }
                walk(child, conditions + [condition])

        walk(self.tree_, [])
        return rules

    def set_feature_names(self, feature_names: Sequence[str]) -> _BaseAdaptiveRangeTree:
        """Assign display names after fitting on an unnamed NumPy matrix."""
        self._check_is_fitted()
        if len(feature_names) != self.n_features_in_:
            raise ValueError("feature_names must match the number of fitted features.")
        self.display_feature_names_ = np.asarray(
            [str(name) for name in feature_names], dtype=object
        )
        for node in self._iter_nodes():
            if not node.is_leaf:
                node.feature_name = str(self.display_feature_names_[node.feature_index])
        return self

    def plot_tree(
        self,
        *,
        ax: Axes | None = None,
        figsize: tuple[float, float] | None = None,
        decimals: int = 2,
        filled: bool = True,
        feature_names: Sequence[str] | None = None,
        class_names: Sequence[str] | None = None,
        fontsize: int = 9,
        title: str = "Adaptive Range Tree",
        save_path: str | None = None,
        dpi: int = 180,
        show: bool = False,
    ) -> tuple[Figure, Axes]:
        """Method form of :func:`plot_tree`."""
        return plot_tree(
            self,
            ax=ax,
            figsize=figsize,
            decimals=decimals,
            filled=filled,
            feature_names=feature_names,
            class_names=class_names,
            fontsize=fontsize,
            title=title,
            save_path=save_path,
            dpi=dpi,
            show=show,
        )

    # ------------------------------------------------------------------
    # Tree construction
    # ------------------------------------------------------------------
    def _grow_tree(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        indices: np.ndarray,
        depth: int,
    ) -> _TreeNode:
        local_y = y[indices]
        local_w = weights[indices]
        node = self._make_node(local_y, local_w, depth)

        if self._should_stop(node, local_y):
            return node

        split = self._best_split(X[indices], local_y, local_w)
        if split is None or split.raw_gain < self.min_impurity_decrease:
            return node

        node.feature_index = split.feature_index
        node.feature_name = str(self.display_feature_names_[split.feature_index])
        node.thresholds = split.thresholds
        node.missing_child = split.missing_child
        node.gain = split.raw_gain

        for branch_idx in range(int(np.max(split.assignments)) + 1):
            mask = split.assignments == branch_idx
            child_indices = indices[mask]
            if child_indices.size == 0:
                continue
            node.children.append(self._grow_tree(X, y, weights, child_indices, depth=depth + 1))
        return node

    def _make_node(self, y: np.ndarray, weights: np.ndarray, depth: int) -> _TreeNode:
        node_id = self._next_node_id
        self._next_node_id += 1
        impurity = self._impurity(y, weights)
        total_weight = float(np.sum(weights))
        if self._task == "classification":
            counts = np.bincount(y, weights=weights, minlength=self.n_classes_).astype(float)
            smoothed_counts = counts + self.probability_smoothing
            probabilities = (
                smoothed_counts / smoothed_counts.sum()
                if smoothed_counts.sum()
                else np.ones(self.n_classes_) / self.n_classes_
            )
            prediction: int | float = int(np.argmax(probabilities))
        else:
            probabilities = None
            prediction = (
                float(np.average(y, weights=weights)) if total_weight else float(np.mean(y))
            )
        return _TreeNode(
            node_id=node_id,
            depth=depth,
            n_samples=len(y),
            weighted_n_samples=total_weight,
            impurity=impurity,
            prediction=prediction,
            probabilities=probabilities,
        )

    def _should_stop(self, node: _TreeNode, y: np.ndarray) -> bool:
        if self.max_depth is not None and node.depth >= self.max_depth:
            return True
        if node.weighted_n_samples < self.min_samples_split:
            return True
        if node.weighted_n_samples < 2 * self.min_samples_leaf:
            return True
        if node.impurity <= np.finfo(float).eps:
            return True
        if self._task == "classification" and np.unique(y).size <= 1:
            return True
        return False

    def _best_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
    ) -> _Split | None:
        parent_impurity = self._impurity(y, weights)
        total_weight = float(np.sum(weights))
        feature_indices = self._sample_features(X.shape[1])
        best: _Split | None = None

        for feature_index in feature_indices:
            candidate = self._best_split_for_feature(
                X[:, feature_index], y, weights, feature_index, parent_impurity, total_weight
            )
            if candidate is None:
                continue
            if best is None or candidate.penalized_gain > best.penalized_gain + 1e-15:
                best = candidate
            elif (
                best is not None
                and abs(candidate.penalized_gain - best.penalized_gain) <= 1e-15
                and candidate.raw_gain > best.raw_gain
            ):
                best = candidate

        if best is None or best.penalized_gain <= 0.0:
            return None
        return best

    def _best_split_for_feature(
        self,
        x: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        feature_index: int,
        parent_impurity: float,
        total_weight: float,
    ) -> _Split | None:
        missing_mask = np.isnan(x)
        observed_mask = ~missing_mask
        if float(np.sum(weights[observed_mask])) < 2 * self.min_samples_leaf:
            return None

        observed_x = x[observed_mask]
        unique_values = np.unique(observed_x)
        if unique_values.size < 2:
            return None

        all_midpoints = unique_values[:-1] + (unique_values[1:] - unique_values[:-1]) / 2.0
        if all_midpoints.size > self.max_bins - 1:
            selected = np.linspace(0, all_midpoints.size - 1, self.max_bins - 1)
            selected = np.unique(np.rint(selected).astype(int))
            base_thresholds = all_midpoints[selected]
        else:
            base_thresholds = all_midpoints

        atomic_assignments = np.searchsorted(base_thresholds, observed_x, side="right")
        n_atomic = len(base_thresholds) + 1
        if n_atomic < 2:
            return None

        observed_y = y[observed_mask]
        observed_w = weights[observed_mask]
        counts = np.bincount(atomic_assignments, weights=observed_w, minlength=n_atomic).astype(
            float
        )
        prefix_counts = np.concatenate(([0], np.cumsum(counts)))

        if self._task == "classification":
            class_matrix = np.zeros((n_atomic, self.n_classes_), dtype=float)
            np.add.at(class_matrix, (atomic_assignments, observed_y), observed_w)
            prefix_stats = np.vstack(
                [np.zeros((1, self.n_classes_), dtype=float), np.cumsum(class_matrix, axis=0)]
            )
            prefix_sum = prefix_sumsq = None
        else:
            weighted_sum = np.bincount(
                atomic_assignments, weights=observed_w * observed_y, minlength=n_atomic
            )
            weighted_sumsq = np.bincount(
                atomic_assignments, weights=observed_w * observed_y * observed_y, minlength=n_atomic
            )
            bin_weight = np.bincount(atomic_assignments, weights=observed_w, minlength=n_atomic)
            prefix_stats = np.concatenate(([0.0], np.cumsum(bin_weight)))
            prefix_sum = np.concatenate(([0.0], np.cumsum(weighted_sum)))
            prefix_sumsq = np.concatenate(([0.0], np.cumsum(weighted_sumsq)))

        def segment_loss(left: int, right: int) -> float:
            if prefix_counts[right] - prefix_counts[left] < self.min_samples_leaf:
                return math.inf
            if self._task == "classification":
                class_counts = prefix_stats[right] - prefix_stats[left]
                segment_weight = float(np.sum(class_counts))
                if segment_weight <= 0:
                    return math.inf
                probabilities = class_counts / segment_weight
                if self.criterion_ == "gini":
                    impurity = 1.0 - float(np.sum(probabilities * probabilities))
                else:
                    positive = probabilities[probabilities > 0]
                    impurity = -float(np.sum(positive * np.log2(positive)))
                return segment_weight * impurity
            segment_weight = float(prefix_stats[right] - prefix_stats[left])
            if segment_weight <= 0:
                return math.inf
            total = float(prefix_sum[right] - prefix_sum[left])
            total_sq = float(prefix_sumsq[right] - prefix_sumsq[left])
            return max(0.0, total_sq - total * total / segment_weight)

        max_k = min(self.max_branches, n_atomic)
        dp = np.full((max_k + 1, n_atomic + 1), np.inf, dtype=float)
        previous = np.full((max_k + 1, n_atomic + 1), -1, dtype=int)
        dp[0, 0] = 0.0

        for k in range(1, max_k + 1):
            for right in range(k, n_atomic + 1):
                for left in range(k - 1, right):
                    if not np.isfinite(dp[k - 1, left]):
                        continue
                    loss = segment_loss(left, right)
                    value = dp[k - 1, left] + loss
                    if value < dp[k, right]:
                        dp[k, right] = value
                        previous[k, right] = left

        missing_loss_separate = 0.0
        can_separate_missing = (
            self.missing_strategy == "separate"
            and float(np.sum(weights[missing_mask])) >= self.min_samples_leaf
        )
        if can_separate_missing:
            missing_loss_separate = float(np.sum(weights[missing_mask])) * self._impurity(
                y[missing_mask], weights[missing_mask]
            )

        best_result: _Split | None = None
        for k in range(2, max_k + 1):
            if not np.isfinite(dp[k, n_atomic]):
                continue

            boundaries: list[int] = []
            right = n_atomic
            valid = True
            for level in range(k, 0, -1):
                left = int(previous[level, right])
                if left < 0:
                    valid = False
                    break
                if level > 1:
                    boundaries.append(left)
                right = left
            if not valid:
                continue
            boundaries.sort()
            thresholds = np.asarray([base_thresholds[b - 1] for b in boundaries], dtype=float)
            assignments = np.searchsorted(thresholds, x, side="right").astype(int)

            numeric_counts = np.bincount(
                assignments[observed_mask], weights=observed_w, minlength=k
            ).astype(float)
            if np.any(numeric_counts < self.min_samples_leaf):
                continue

            child_loss = float(dp[k, n_atomic])
            if can_separate_missing:
                missing_child = k
                assignments[missing_mask] = missing_child
                child_loss += missing_loss_separate
            else:
                missing_child = int(np.argmax(numeric_counts))
                assignments[missing_mask] = missing_child
                if np.any(missing_mask):
                    # Recompute exact child loss after merging missing values.
                    child_loss = 0.0
                    for branch_idx in range(k):
                        mask = assignments == branch_idx
                        branch_weight = float(np.sum(weights[mask]))
                        child_loss += branch_weight * self._impurity(y[mask], weights[mask])

            raw_gain = parent_impurity - child_loss / total_weight
            gain_for_scoring = raw_gain
            if self.normalize_gain:
                gain_for_scoring = raw_gain / max(abs(parent_impurity), np.finfo(float).eps)
            penalized_gain = gain_for_scoring - self.branch_penalty * (k - 1)
            result = _Split(
                feature_index=feature_index,
                thresholds=thresholds,
                assignments=assignments,
                missing_child=missing_child,
                raw_gain=float(raw_gain),
                penalized_gain=float(penalized_gain),
            )
            if best_result is None or result.penalized_gain > best_result.penalized_gain:
                best_result = result

        return best_result

    # ------------------------------------------------------------------
    # Prediction and traversal helpers
    # ------------------------------------------------------------------
    def _leaf_for_row(self, row: np.ndarray) -> _TreeNode:
        leaf, _, _ = self._trace_row(row)
        return leaf

    def _trace_row(self, row: np.ndarray) -> tuple[_TreeNode, list[_TreeNode], list[int]]:
        node = self.tree_
        nodes = [node]
        branch_indices: list[int] = []
        while not node.is_leaf:
            value = row[node.feature_index]
            if np.isnan(value):
                child_index = int(node.missing_child)
            else:
                child_index = int(np.searchsorted(node.thresholds, value, side="right"))
            # Defensive fallback for serialized or user-modified trees.
            child_index = min(max(child_index, 0), len(node.children) - 1)
            branch_indices.append(child_index)
            node = node.children[child_index]
            nodes.append(node)
        return node, nodes, branch_indices

    # ------------------------------------------------------------------
    # Statistics and validation
    # ------------------------------------------------------------------
    def _impurity(self, y: np.ndarray, weights: np.ndarray) -> float:
        total_weight = float(np.sum(weights))
        if y.size == 0 or total_weight <= 0:
            return 0.0
        if self._task == "classification":
            counts = np.bincount(y, weights=weights, minlength=self.n_classes_).astype(float)
            probabilities = counts / counts.sum()
            if self.criterion_ == "gini":
                return max(0.0, 1.0 - float(np.sum(probabilities * probabilities)))
            positive = probabilities[probabilities > 0]
            return -float(np.sum(positive * np.log2(positive)))
        mean = float(np.average(y, weights=weights))
        return float(np.average((y - mean) ** 2, weights=weights))

    def _sample_features(self, n_features: int) -> np.ndarray:
        max_features = self.max_features
        if max_features is None:
            count = n_features
        elif isinstance(max_features, str):
            if max_features == "sqrt":
                count = max(1, int(math.sqrt(n_features)))
            elif max_features == "log2":
                count = max(1, int(math.log2(n_features)))
            else:
                raise ValueError("max_features string must be 'sqrt' or 'log2'.")
        elif isinstance(max_features, numbers.Integral):
            count = int(max_features)
        elif isinstance(max_features, numbers.Real):
            if not 0 < float(max_features) <= 1:
                raise ValueError("Float max_features must be in (0, 1].")
            count = max(1, int(math.ceil(float(max_features) * n_features)))
        else:
            raise ValueError("Unsupported max_features value.")
        if not 1 <= count <= n_features:
            raise ValueError("max_features selects an invalid number of features.")
        if count == n_features:
            return np.arange(n_features)
        return np.sort(self._rng.choice(n_features, size=count, replace=False))

    def _prepare_X(self, X: ArrayLike, fitting: bool) -> tuple[np.ndarray, list[str]]:
        if hasattr(X, "columns") and hasattr(X, "to_numpy"):
            feature_names = [str(column) for column in X.columns]
        else:
            raw = np.asarray(X)
            width = raw.shape[1] if raw.ndim == 2 else 1
            feature_names = [f"x{i}" for i in range(width)]
        try:
            array = validate_data(
                self,
                X,
                dtype=float,
                ensure_2d=True,
                ensure_all_finite="allow-nan",
                reset=fitting,
            )
        except (TypeError, ValueError) as exc:
            message = str(exc)
            if "could not convert" in message.lower() or "dtype" in message.lower():
                raise ValueError("All features must currently be numeric.") from exc
            raise
        return array, feature_names

    @staticmethod
    def _prepare_sample_weight(sample_weight: Sequence[float] | None, n_samples: int) -> np.ndarray:
        if sample_weight is None:
            return np.ones(n_samples, dtype=float)
        weights = np.asarray(sample_weight, dtype=float)
        if weights.ndim != 1 or len(weights) != n_samples:
            raise ValueError("sample_weight must contain one value per training row.")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("sample_weight must contain finite, non-negative values.")
        if np.sum(weights) <= 0:
            raise ValueError("sample_weight must have a positive total.")
        return weights

    def _validate_hyperparameters(self) -> None:
        if self.max_depth is not None and self.max_depth < 0:
            raise ValueError("max_depth must be non-negative or None.")
        if self.max_branches < 2:
            raise ValueError("max_branches must be at least 2.")
        if self.max_bins < 2:
            raise ValueError("max_bins must be at least 2.")
        if self.min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2.")
        if self.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be at least 1.")
        if self.branch_penalty < 0 or self.min_impurity_decrease < 0:
            raise ValueError("Penalty and impurity-decrease parameters cannot be negative.")
        if self.probability_smoothing < 0:
            raise ValueError("probability_smoothing cannot be negative.")
        if not isinstance(self.normalize_gain, (bool, np.bool_)):
            raise ValueError("normalize_gain must be True or False.")
        if self.missing_strategy not in {"separate", "largest"}:
            raise ValueError("missing_strategy must be 'separate' or 'largest'.")

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "tree_"):
            raise NotFittedError("Call fit before using this estimator.")

    def _iter_nodes(self) -> Iterable[_TreeNode]:
        stack = [self.tree_]
        while stack:
            node = stack.pop()
            yield node
            stack.extend(reversed(node.children))

    def _display_prediction(self, node: _TreeNode) -> Any:
        if self._task == "classification":
            value = self.classes_[int(node.prediction)]
            return value.item() if hasattr(value, "item") else value
        return float(node.prediction)

    def _leaf_text(self, node: _TreeNode, decimals: int) -> str:
        if self._task == "classification":
            probability = float(node.probabilities[int(node.prediction)])
            return f"class={self._display_prediction(node)!r}, p={probability:.{decimals}f}"
        return f"value={float(node.prediction):.{decimals}f}"

    @staticmethod
    def _format_number(value: float, decimals: int) -> str:
        return f"{value:.{decimals}f}".rstrip("0").rstrip(".")

    def _branch_label(self, node: _TreeNode, child_index: int, decimals: int = 3) -> str:
        if child_index == node.missing_child and child_index >= len(node.thresholds) + 1:
            return "missing"
        thresholds = node.thresholds
        if child_index == 0:
            return f"< {self._format_number(float(thresholds[0]), decimals)}"
        if child_index == len(thresholds):
            return f">= {self._format_number(float(thresholds[-1]), decimals)}"
        lower = self._format_number(float(thresholds[child_index - 1]), decimals)
        upper = self._format_number(float(thresholds[child_index]), decimals)
        return f"[{lower}, {upper})"


class AdaptiveRangeTreeClassifier(ClassifierMixin, _BaseAdaptiveRangeTree):
    """Adaptive multi-range decision-tree classifier."""

    _task = "classification"

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """Return leaf class-probability estimates."""
        self._check_is_fitted()
        X_array, _ = self._prepare_X(X, fitting=False)
        return np.vstack([self._leaf_for_row(row).probabilities for row in X_array])

    def predict_log_proba(self, X: ArrayLike) -> np.ndarray:
        """Return log probabilities, using the standard log(0)=-inf convention."""
        with np.errstate(divide="ignore"):
            return np.log(self.predict_proba(X))


class AdaptiveRangeTreeRegressor(RegressorMixin, _BaseAdaptiveRangeTree):
    """Adaptive multi-range decision-tree regressor."""

    _task = "regression"


def plot_tree(
    estimator: _BaseAdaptiveRangeTree,
    *,
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
    decimals: int = 2,
    filled: bool = True,
    feature_names: Sequence[str] | None = None,
    class_names: Sequence[str] | None = None,
    fontsize: int = 9,
    title: str = "Adaptive Range Tree",
    save_path: str | None = None,
    dpi: int = 180,
    show: bool = False,
) -> tuple[Figure, Axes]:
    """Plot a fitted adaptive range tree.

    The layout resembles ``sklearn.tree.plot_tree``, while edge labels show the
    several intervals emitted by each multiway node.

    Returns
    -------
    (figure, axes)
        Matplotlib objects for further customization.
    """
    estimator._check_is_fitted()
    if feature_names is not None and len(feature_names) != estimator.n_features_in_:
        raise ValueError("feature_names must match the number of fitted features.")
    if class_names is not None and estimator._task != "classification":
        raise ValueError("class_names is only valid for classification trees.")
    if class_names is not None and len(class_names) != estimator.n_classes_:
        raise ValueError("class_names must match the number of fitted classes.")

    leaves = max(1, estimator.n_leaves_)
    if figsize is None:
        figsize = (max(12.0, min(32.0, 2.6 * leaves)), max(6.0, 3.2 * (estimator.max_depth_ + 1)))
    if ax is None:
        figure, ax = plt.subplots(figsize=figsize)
    else:
        figure = ax.figure

    x_positions: dict[int, float] = {}
    y_positions: dict[int, float] = {}
    next_leaf_x = 0.0

    def assign_positions(node: _TreeNode) -> float:
        nonlocal next_leaf_x
        y_positions[node.node_id] = -float(node.depth)
        if node.is_leaf:
            x_positions[node.node_id] = next_leaf_x
            next_leaf_x += 1.0
        else:
            child_x = [assign_positions(child) for child in node.children]
            x_positions[node.node_id] = float(np.mean(child_x))
        return x_positions[node.node_id]

    assign_positions(estimator.tree_)

    if estimator._task == "classification":
        palette = plt.get_cmap("Pastel1")

    display_features = (
        [str(value) for value in feature_names]
        if feature_names is not None
        else [str(value) for value in estimator.display_feature_names_]
    )

    def node_label(node: _TreeNode) -> str:
        if node.is_leaf:
            if estimator._task == "classification":
                class_index = int(node.prediction)
                predicted_class = (
                    str(class_names[class_index])
                    if class_names is not None
                    else str(estimator._display_prediction(node))
                )
                probability = float(node.probabilities[class_index])
                return (
                    f"leaf #{node.node_id}\nclass = {predicted_class}\n"
                    f"p = {probability:.{decimals}f}\n"
                    f"n = {node.n_samples}\nimpurity = {node.impurity:.{decimals}f}"
                )
            return (
                f"leaf #{node.node_id}\nvalue = {float(node.prediction):.{decimals}f}\n"
                f"n = {node.n_samples}\nimpurity = {node.impurity:.{decimals}f}"
            )
        feature = display_features[node.feature_index]
        threshold_text = ", ".join(
            estimator._format_number(float(value), decimals) for value in node.thresholds
        )
        return (
            f"{feature}\ncut points = [{threshold_text}]\n"
            f"gain = {node.gain:.{decimals}f}\n"
            f"n = {node.n_samples}\nimpurity = {node.impurity:.{decimals}f}"
        )

    def draw(node: _TreeNode) -> None:
        x = x_positions[node.node_id]
        y = y_positions[node.node_id]
        if filled:
            if estimator._task == "classification":
                color = palette(int(node.prediction) % palette.N)
            else:
                all_predictions = [
                    float(n.prediction) for n in estimator._iter_nodes() if n.is_leaf
                ]
                low, high = min(all_predictions), max(all_predictions)
                scaled = 0.5 if high == low else (float(node.prediction) - low) / (high - low)
                color = plt.get_cmap("Blues")(0.20 + 0.55 * scaled)
        else:
            color = "white"

        ax.text(
            x,
            y,
            node_label(node),
            ha="center",
            va="center",
            fontsize=fontsize,
            bbox={
                "boxstyle": "round,pad=0.45",
                "facecolor": color,
                "edgecolor": "#34495e",
                "linewidth": 1.25,
            },
            zorder=3,
        )

        for child_index, child in enumerate(node.children):
            child_x = x_positions[child.node_id]
            child_y = y_positions[child.node_id]
            ax.annotate(
                "",
                xy=(child_x, child_y + 0.14),
                xytext=(x, y - 0.14),
                arrowprops={"arrowstyle": "-", "color": "#607d8b", "lw": 1.25},
                zorder=1,
            )
            midpoint_x = (x + child_x) / 2.0
            midpoint_y = (y + child_y) / 2.0
            ax.text(
                midpoint_x,
                midpoint_y,
                estimator._branch_label(node, child_index, decimals),
                ha="center",
                va="center",
                fontsize=max(7, fontsize - 1),
                color="#263238",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.5},
                zorder=2,
            )
            draw(child)

    draw(estimator.tree_)
    ax.set_xlim(-0.75, max(0.25, leaves - 0.25))
    ax.set_ylim(-estimator.max_depth_ - 0.65, 0.65)
    ax.set_title(title, fontsize=fontsize + 3, pad=14)
    ax.axis("off")
    figure.tight_layout()
    if save_path:
        figure.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    return figure, ax


# Compatibility aliases for the original MVP API. They intentionally point to
# the same implementations; statistical stability regularization is not yet
# claimed by the public package API.
StableAdaptiveRangeTreeClassifier = AdaptiveRangeTreeClassifier
StableAdaptiveRangeTreeRegressor = AdaptiveRangeTreeRegressor


__all__ = [
    "AdaptiveRangeTreeClassifier",
    "AdaptiveRangeTreeRegressor",
    "StableAdaptiveRangeTreeClassifier",
    "StableAdaptiveRangeTreeRegressor",
    "plot_tree",
]
