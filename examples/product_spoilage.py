"""End-to-end usage example for the Adaptive Range Tree package.

This script:
1. Generates a reproducible synthetic product dataset.
2. Creates a binary spoilage target and continuous spoilage-loss target.
3. Trains classification and regression Adaptive Range Trees.
4. Evaluates both models on held-out data.
5. Prints learned range rules and one prediction explanation.
6. Saves the dataset, rules, metrics, and tree visualizations.

Run from the repository root after installing the package:

    python examples/product_spoilage.py

Optional arguments:

    python examples/product_spoilage.py --samples 10000 --seed 42 \
        --output-dir product_results --show-plots

Dependencies:

    pip install numpy pandas scikit-learn matplotlib

Important MVP limitation
------------------------
``product_category`` and ``product_subcategory`` are integer identifiers.
The current tree treats every numeric feature as ordered, so category IDs may
be split into ranges. That is acceptable for exercising this numeric-only MVP,
but production categorical support should use native category subsets or an
intentional encoding strategy rather than assuming the identifiers are ordinal.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from rangebranch import (
    AdaptiveRangeTreeClassifier,
    AdaptiveRangeTreeRegressor,
    plot_tree,
)

FEATURE_COLUMNS = [
    "product_weight",
    "product_cost",
    "product_category",
    "product_subcategory",
    "height",
    "width",
    "length",
    "quantity",
    "cogs",
    "retail_price",
    "shelf_life_days",
    "warehouse_dwell_days",
    "average_temperature",
    "temperature_excursion_hours",
    "relative_humidity",
    "distance_miles",
    "handling_events",
]


def _sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-value))


def generate_synthetic_product_data(
    n_samples: int = 6_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate product features with known nonlinear spoilage relationships.

    The data is synthetic, but the variables are linked coherently:

    * category influences weight, dimensions, cost, shelf life, and risk;
    * subcategory is nested within category;
    * COGS and retail price are related but not identical;
    * temperature risk depends on whether the category is temperature-sensitive;
    * dwell time becomes especially dangerous near the end of shelf life;
    * weight has a deliberately non-monotonic range effect;
    * spoilage cost depends on both probability and economic exposure.
    """
    if n_samples < 500:
        raise ValueError("Use at least 500 samples so each learned range has support.")

    rng = np.random.default_rng(random_state)

    # Eight numeric category IDs, each containing five nested subcategories.
    category_probabilities = np.array([0.15, 0.14, 0.13, 0.12, 0.13, 0.12, 0.11, 0.10])
    product_category = rng.choice(8, size=n_samples, p=category_probabilities)
    local_subcategory = rng.integers(0, 5, size=n_samples)
    product_subcategory = product_category * 5 + local_subcategory

    category_weight_scale = np.array([8, 18, 28, 12, 42, 65, 24, 95], dtype=float)
    product_weight = rng.lognormal(mean=np.log(category_weight_scale[product_category]), sigma=0.42)
    product_weight = np.clip(product_weight, 1.0, 220.0)

    # Dimensions are correlated with weight, but shape varies by subcategory.
    cube_root_weight = np.cbrt(product_weight)
    shape_factor = 0.82 + 0.10 * local_subcategory
    height = np.clip(
        1.6 * cube_root_weight * shape_factor + rng.normal(0, 0.7, n_samples), 0.5, None
    )
    width = np.clip(
        2.0 * cube_root_weight / shape_factor + rng.normal(0, 0.8, n_samples), 0.5, None
    )
    length = np.clip(
        2.5 * cube_root_weight + 0.35 * local_subcategory + rng.normal(0, 1.0, n_samples), 0.5, None
    )

    quantity = rng.choice(
        np.array([1, 2, 3, 4, 6, 8, 12, 18, 24]),
        size=n_samples,
        p=np.array([0.23, 0.18, 0.13, 0.12, 0.11, 0.08, 0.07, 0.04, 0.04]),
    )

    category_cost_multiplier = np.array([0.16, 0.21, 0.27, 0.30, 0.19, 0.24, 0.36, 0.14])
    subcategory_cost_multiplier = 0.88 + 0.07 * local_subcategory
    cogs = (
        0.75
        + product_weight * category_cost_multiplier[product_category] * subcategory_cost_multiplier
        + rng.gamma(2.0, 0.55, n_samples)
    )
    cogs = np.clip(cogs, 0.50, None)

    gross_margin_rate = np.clip(
        0.22 + 0.025 * product_category + rng.normal(0, 0.055, n_samples),
        0.10,
        0.58,
    )
    product_cost = cogs / (1.0 - gross_margin_rate)
    retail_price = product_cost * rng.normal(1.08, 0.035, n_samples)

    category_shelf_life = np.array([7, 10, 16, 24, 45, 75, 120, 210], dtype=float)
    shelf_life_days = np.maximum(
        3,
        np.rint(
            category_shelf_life[product_category]
            * rng.lognormal(mean=0.0, sigma=0.20, size=n_samples)
        ),
    ).astype(int)

    warehouse_dwell_days = np.maximum(
        0,
        np.rint(
            rng.gamma(shape=2.1, scale=3.2, size=n_samples)
            + 0.30 * local_subcategory
            + 0.025 * product_weight
        ),
    ).astype(int)

    temperature_sensitive = np.isin(product_category, [0, 1, 2, 3])
    average_temperature = np.where(
        temperature_sensitive,
        rng.normal(38.0, 4.2, n_samples),
        rng.normal(68.0, 7.5, n_samples),
    )
    excursion_rate = np.where(temperature_sensitive, 1.8, 0.7)
    temperature_excursion_hours = rng.gamma(1.4, excursion_rate, n_samples)
    relative_humidity = np.clip(
        rng.normal(55 + 3.5 * np.isin(product_category, [0, 3, 4]), 12, n_samples),
        18,
        95,
    )

    distance_miles = np.clip(rng.gamma(2.2, 190.0, n_samples), 5, 2_500)
    handling_events = np.maximum(
        1,
        rng.poisson(2.5 + distance_miles / 420 + quantity / 14, n_samples),
    )

    # ------------------------------------------------------------------
    # Known spoilage-generating process.
    # It intentionally contains thresholds, interactions, and a non-monotonic
    # weight effect so a multi-range tree has meaningful structure to recover.
    # ------------------------------------------------------------------
    category_log_odds = np.array([1.05, 0.75, 0.40, 0.15, -0.20, -0.45, -0.70, -0.95])
    subcategory_risk = np.array([-0.25, 0.05, 0.30, -0.10, 0.45])[local_subcategory]
    shelf_utilization = warehouse_dwell_days / np.maximum(shelf_life_days, 1)

    weight_range_effect = np.select(
        [
            product_weight < 10,
            product_weight < 35,
            product_weight < 75,
            product_weight < 125,
        ],
        [-0.35, 0.20, 0.72, 0.10],
        default=0.55,
    )

    cold_temperature_risk = temperature_sensitive * np.select(
        [average_temperature < 34, average_temperature < 39, average_temperature < 44],
        [0.15, -0.20, 0.65],
        default=1.35,
    )
    ambient_temperature_risk = (~temperature_sensitive) * (average_temperature > 78) * 0.40

    log_odds = (
        -3.00
        + category_log_odds[product_category]
        + subcategory_risk
        + weight_range_effect
        + 2.8 * np.maximum(shelf_utilization - 0.45, 0)
        + 1.5 * (shelf_utilization > 0.80)
        + cold_temperature_risk
        + ambient_temperature_risk
        + 0.12 * np.maximum(temperature_excursion_hours - 2.0, 0)
        + 0.018 * np.maximum(relative_humidity - 70, 0)
        + 0.10 * np.maximum(handling_events - 5, 0)
        + 0.25 * (quantity >= 12)
        + 0.00020 * np.maximum(distance_miles - 600, 0)
        + rng.normal(0, 0.30, n_samples)
    )

    spoilage_probability = np.clip(_sigmoid(log_odds), 0.005, 0.97)
    spoiled = rng.binomial(1, spoilage_probability).astype(int)

    economic_exposure = cogs * quantity
    spoilage_fraction = np.clip(
        0.04 + 0.72 * spoilage_probability + 0.16 * spoiled + rng.normal(0, 0.055, n_samples),
        0,
        1,
    )
    spoilage_loss = np.maximum(
        0,
        economic_exposure * spoilage_fraction + rng.normal(0, 0.35, n_samples),
    )

    frame = pd.DataFrame(
        {
            "product_weight": product_weight,
            "product_cost": product_cost,
            "product_category": product_category,
            "product_subcategory": product_subcategory,
            "height": height,
            "width": width,
            "length": length,
            "quantity": quantity,
            "cogs": cogs,
            "retail_price": retail_price,
            "shelf_life_days": shelf_life_days,
            "warehouse_dwell_days": warehouse_dwell_days,
            "average_temperature": average_temperature,
            "temperature_excursion_hours": temperature_excursion_hours,
            "relative_humidity": relative_humidity,
            "distance_miles": distance_miles,
            "handling_events": handling_events,
            "spoilage_probability_true": spoilage_probability,
            "spoiled": spoiled,
            "spoilage_loss": spoilage_loss,
        }
    )

    # Add a few missing measurements to exercise the model's missing routing.
    for column, missing_rate in {
        "product_weight": 0.012,
        "average_temperature": 0.018,
        "relative_humidity": 0.015,
        "distance_miles": 0.008,
    }.items():
        missing_rows = rng.random(n_samples) < missing_rate
        frame.loc[missing_rows, column] = np.nan

    return frame


def train_product_trees(
    data: pd.DataFrame,
    random_state: int = 42,
) -> tuple[
    AdaptiveRangeTreeClassifier,
    AdaptiveRangeTreeRegressor,
    dict[str, Any],
    pd.DataFrame,
]:
    """Split the dataset, train both estimators, and calculate test metrics."""
    X = data[FEATURE_COLUMNS]
    y_classification = data["spoiled"]
    y_regression = data["spoilage_loss"]

    train_index, test_index = train_test_split(
        np.arange(len(data)),
        test_size=0.25,
        random_state=random_state,
        stratify=y_classification,
    )

    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]
    y_class_train = y_classification.iloc[train_index]
    y_class_test = y_classification.iloc[test_index]
    y_reg_train = y_regression.iloc[train_index]
    y_reg_test = y_regression.iloc[test_index]

    classifier = AdaptiveRangeTreeClassifier(
        max_depth=4,
        max_branches=4,
        max_bins=40,
        min_samples_split=100,
        min_samples_leaf=45,
        min_impurity_decrease=0.001,
        branch_penalty=0.003,
        criterion="log_loss",
        missing_strategy="separate",
        random_state=random_state,
    )
    classifier.fit(X_train, y_class_train)

    class_prediction = classifier.predict(X_test)
    class_probability_matrix = classifier.predict_proba(X_test)
    positive_class_index = int(np.flatnonzero(classifier.classes_ == 1)[0])
    class_probability = class_probability_matrix[:, positive_class_index]

    regression = AdaptiveRangeTreeRegressor(
        max_depth=4,
        max_branches=4,
        max_bins=40,
        min_samples_split=100,
        min_samples_leaf=45,
        min_impurity_decrease=0.01,
        branch_penalty=0.02,
        criterion="squared_error",
        missing_strategy="separate",
        random_state=random_state,
    )
    regression.fit(X_train, y_reg_train)
    regression_prediction = regression.predict(X_test)

    metrics: dict[str, Any] = {
        "dataset": {
            "rows": int(len(data)),
            "training_rows": int(len(train_index)),
            "test_rows": int(len(test_index)),
            "features": len(FEATURE_COLUMNS),
            "spoilage_rate": float(y_classification.mean()),
        },
        "classifier": {
            "accuracy": float(accuracy_score(y_class_test, class_prediction)),
            "balanced_accuracy": float(balanced_accuracy_score(y_class_test, class_prediction)),
            "roc_auc": float(roc_auc_score(y_class_test, class_probability)),
            "average_precision": float(average_precision_score(y_class_test, class_probability)),
            "log_loss": float(log_loss(y_class_test, class_probability_matrix)),
            "confusion_matrix": confusion_matrix(y_class_test, class_prediction).tolist(),
            "nodes": int(classifier.n_nodes_),
            "leaves": int(classifier.n_leaves_),
            "depth": int(classifier.max_depth_),
        },
        "regressor": {
            "mae": float(mean_absolute_error(y_reg_test, regression_prediction)),
            "rmse": float(np.sqrt(mean_squared_error(y_reg_test, regression_prediction))),
            "r2": float(r2_score(y_reg_test, regression_prediction)),
            "nodes": int(regression.n_nodes_),
            "leaves": int(regression.n_leaves_),
            "depth": int(regression.max_depth_),
        },
    }

    scored_test = data.iloc[test_index].copy()
    scored_test["predicted_spoiled"] = class_prediction
    scored_test["predicted_spoilage_probability"] = class_probability
    scored_test["predicted_spoilage_loss"] = regression_prediction

    print("\nCLASSIFICATION REPORT")
    print(classification_report(y_class_test, class_prediction, digits=3))
    return classifier, regression, metrics, scored_test


def save_results(
    data: pd.DataFrame,
    scored_test: pd.DataFrame,
    classifier: AdaptiveRangeTreeClassifier,
    regression: AdaptiveRangeTreeRegressor,
    metrics: dict[str, Any],
    output_dir: Path,
    show_plots: bool = False,
) -> None:
    """Save generated data, metrics, rules, text trees, and PNG plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    data.to_csv(output_dir / "synthetic_product_data.csv", index=False)
    scored_test.to_csv(output_dir / "scored_test_data.csv", index=False)

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    with (output_dir / "classification_rules.json").open("w", encoding="utf-8") as file:
        json.dump(classifier.get_rules(), file, indent=2)
    with (output_dir / "regression_rules.json").open("w", encoding="utf-8") as file:
        json.dump(regression.get_rules(), file, indent=2)

    (output_dir / "classification_tree.txt").write_text(classifier.export_text(), encoding="utf-8")
    (output_dir / "regression_tree.txt").write_text(regression.export_text(), encoding="utf-8")

    classifier_figure, _ = plot_tree(
        classifier,
        title="Synthetic Product Spoilage Classifier",
        save_path=str(output_dir / "classification_tree.png"),
        dpi=200,
        show=show_plots,
    )
    plt.close(classifier_figure)
    regression_figure, _ = plot_tree(
        regression,
        title="Synthetic Product Spoilage-Loss Regressor",
        save_path=str(output_dir / "regression_tree.png"),
        dpi=200,
        show=show_plots,
    )
    plt.close(regression_figure)


def run_example(
    n_samples: int = 6_000,
    random_state: int = 42,
    output_dir: str = "sart_product_results",
    show_plots: bool = False,
) -> dict[str, Any]:
    """Execute the complete product-data demonstration."""
    destination = Path(output_dir)
    data = generate_synthetic_product_data(
        n_samples=n_samples,
        random_state=random_state,
    )

    print("SYNTHETIC PRODUCT DATASET")
    print(f"Rows: {len(data):,}")
    print(f"Features: {len(FEATURE_COLUMNS)}")
    print(f"Spoilage rate: {data['spoiled'].mean():.2%}")
    print("\nFirst five rows:")
    print(data.head().to_string(index=False))

    classifier, regression, metrics, scored_test = train_product_trees(
        data,
        random_state=random_state,
    )

    print("\nMODEL METRICS")
    print(json.dumps(metrics, indent=2))

    print("\nCLASSIFICATION TREE")
    print(classifier.export_text())
    print("\nREGRESSION TREE")
    print(regression.export_text())

    example_row = scored_test.iloc[0][FEATURE_COLUMNS]
    print("\nEXAMPLE CLASSIFICATION EXPLANATION")
    print(json.dumps(classifier.explain_prediction(example_row), indent=2))
    print("\nEXAMPLE REGRESSION EXPLANATION")
    print(json.dumps(regression.explain_prediction(example_row), indent=2))

    save_results(
        data=data,
        scored_test=scored_test,
        classifier=classifier,
        regression=regression,
        metrics=metrics,
        output_dir=destination,
        show_plots=show_plots,
    )

    print(f"\nSaved outputs to: {destination.resolve()}")
    return metrics


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate product data and train Stable Adaptive Range Trees."
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=6_000,
        help="Number of synthetic product rows to generate (default: 6000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for generation and splitting (default: 42).",
    )
    parser.add_argument(
        "--output-dir",
        default="sart_product_results",
        help="Directory for generated CSV, JSON, TXT, and PNG files.",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display plots interactively in addition to saving them.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    run_example(
        n_samples=arguments.samples,
        random_state=arguments.seed,
        output_dir=arguments.output_dir,
        show_plots=arguments.show_plots,
    )
