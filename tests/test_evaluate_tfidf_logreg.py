from __future__ import annotations

import numpy as np
import pytest

from scripts.evaluate_tfidf_logreg import (
    _calculate_metrics_by_threshold,
    _predict_scores,
    _train_vectorizer_and_model,
    evaluate_configuration,
)


def test_train_vectorizer_and_model_predicts_probability_scores() -> None:
    train_comments = [
        "좋은 댓글",
        "오늘 날씨 좋다",
        "ㅅㅂ 꺼져",
        "병신 같은 소리",
    ]
    train_labels = np.asarray([0, 0, 1, 1], dtype=np.float32)

    vectorizer, model = _train_vectorizer_and_model(
        train_comments=train_comments,
        train_labels=train_labels,
        ngram_min=2,
        ngram_max=3,
        c_value=1.0,
        random_state=42,
        use_class_weight=False,
    )
    scores = _predict_scores(model, vectorizer, ["좋은 하루", "ㅅㅂ 뭐래"])

    assert scores.shape == (2,)
    assert scores.dtype == np.float32
    assert np.all((scores >= 0.0) & (scores <= 1.0))


def test_train_vectorizer_and_model_applies_balanced_class_weight() -> None:
    vectorizer, model = _train_vectorizer_and_model(
        train_comments=["좋다", "나쁘다", "병신"],
        train_labels=np.asarray([0, 0, 1], dtype=np.float32),
        ngram_min=2,
        ngram_max=3,
        c_value=1.0,
        random_state=42,
        use_class_weight=True,
    )

    assert vectorizer.analyzer == "char"
    assert model.class_weight == "balanced"


def test_calculate_metrics_by_threshold_compares_thresholds() -> None:
    labels = np.asarray([0, 1, 1, 0], dtype=np.float32)
    scores = np.asarray([0.1, 0.4, 0.8, 0.6], dtype=np.float32)

    metrics = _calculate_metrics_by_threshold(labels, scores, [0.5, 0.3])

    assert metrics[0.5]["precision"] == pytest.approx(0.5)
    assert metrics[0.5]["recall"] == pytest.approx(0.5)
    assert metrics[0.3]["precision"] == pytest.approx(2 / 3)
    assert metrics[0.3]["recall"] == pytest.approx(1.0)


def test_evaluate_configuration_returns_metrics_and_settings() -> None:
    result = evaluate_configuration(
        comments=["좋은 댓글", "나쁜 댓글", "ㅅㅂ 꺼져", "병신 같은 말"],
        labels=np.asarray([0, 0, 1, 1], dtype=np.float32),
        val_size=0.5,
        random_state=42,
        threshold=0.5,
        ngram_min=2,
        ngram_max=3,
        c_value=1.0,
        use_class_weight=False,
    )

    assert result["threshold"] == 0.5
    assert result["ngram_range"] == (2, 3)
    assert result["c_value"] == 1.0
    assert result["use_class_weight"] is False
    assert set(result["metrics"]) == {
        "roc_auc",
        "f1",
        "accuracy",
        "precision",
        "recall",
    }
