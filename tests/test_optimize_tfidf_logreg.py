from __future__ import annotations

import numpy as np

from scripts.optimize_tfidf_logreg import _average_metrics, evaluate_configuration_cv


def test_average_metrics_returns_fold_means() -> None:
    averaged = _average_metrics(
        [
            {"f1": 0.8, "roc_auc": 0.9},
            {"f1": 0.6, "roc_auc": 0.7},
        ],
    )

    assert averaged == {
        "f1": 0.7,
        "roc_auc": 0.8,
    }


def test_evaluate_configuration_cv_returns_metric_dictionary() -> None:
    metrics = evaluate_configuration_cv(
        comments=[
            "좋은 댓글",
            "오늘 날씨 좋다",
            "ㅅㅂ 꺼져",
            "병신 같은 소리",
            "정말 고맙다",
            "개같은 말이네",
        ],
        labels=np.asarray([0, 0, 1, 1, 0, 1], dtype=np.float32),
        n_splits=3,
        random_state=42,
        ngram_min=2,
        ngram_max=4,
        c_value=1.0,
        use_class_weight=True,
        threshold=0.5,
    )

    assert set(metrics) == {
        "roc_auc",
        "f1",
        "accuracy",
        "precision",
        "recall",
    }
