import json

import matplotlib.pyplot as plt
import numpy as np

from rangebranch import (
    AdaptiveRangeTreeClassifier,
    explain_prediction,
    export_rules,
    export_text,
    plot_tree,
)


def fitted_model():
    X = np.arange(120, dtype=float).reshape(-1, 1)
    y = np.select([X[:, 0] < 30, X[:, 0] < 80], [0, 1], default=0)
    return AdaptiveRangeTreeClassifier(
        max_depth=2,
        max_branches=3,
        min_samples_leaf=10,
        branch_penalty=0.0,
    ).fit(X, y)


def test_rules_and_explanation_are_json_serializable():
    model = fitted_model()
    rules = export_rules(model)
    explanation = explain_prediction(model, [50.0])

    assert len(rules) == model.n_leaves_
    assert "interval" in rules[0]["conditions"][0]
    assert explanation["decisions"]
    json.dumps(rules)
    json.dumps(explanation)


def test_text_and_plot_exports(tmp_path):
    model = fitted_model()
    text = export_text(model)
    destination = tmp_path / "tree.png"
    figure, axes = plot_tree(model, save_path=str(destination))

    assert "Predict" in text
    assert destination.exists() and destination.stat().st_size > 0
    assert axes.figure is figure
    plt.close(figure)
