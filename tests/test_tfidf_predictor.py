from __future__ import annotations

import json
from pathlib import Path

import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.moderation.tfidf_predictor import TFIDFPredictor, build_model_config, save_pickle


def test_from_artifacts_rejects_missing_vectorizer_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        TFIDFPredictor.from_artifacts(tmp_path)


def test_predict_returns_label_and_score(tmp_path: Path) -> None:
    _save_tfidf_artifacts(tmp_path, threshold=0.47)

    predictor = TFIDFPredictor.from_artifacts(tmp_path)
    result = predictor.predict("ㅅㅂ 뭐래")

    assert set(result) == {"label", "score"}
    assert result["label"] in {"toxic", "non-toxic"}
    assert isinstance(result["score"], float)
    assert 0.0 <= result["score"] <= 1.0


def test_tfidf_predictor_loads_artifacts_and_predicts(tmp_path: Path) -> None:
    _save_tfidf_artifacts(tmp_path, threshold=0.47)

    predictor = TFIDFPredictor.from_artifacts(tmp_path)
    result = predictor.predict("너 진짜 최악이다")

    assert set(result) == {"label", "score"}
    assert result["label"] in {"toxic", "non-toxic"}
    assert isinstance(result["score"], float)
    assert 0.0 <= result["score"] <= 1.0


def test_predict_uses_threshold_for_label(tmp_path: Path) -> None:
    _save_tfidf_artifacts(tmp_path, threshold=0.99)

    predictor = TFIDFPredictor.from_artifacts(tmp_path)
    result = predictor.predict("ㅅㅂ 뭐래")

    assert result["label"] == "non-toxic"


def test_build_model_config_returns_expected_fields() -> None:
    config = build_model_config(
        threshold=0.47,
        ngram_min=2,
        ngram_max=4,
        c_value=2.0,
        use_class_weight=True,
    )

    assert config["model_type"] == "tfidf_logreg"
    assert config["threshold"] == 0.47
    assert config["ngram_min"] == 2
    assert config["ngram_max"] == 4


def _save_tfidf_artifacts(artifact_dir: Path, threshold: float) -> None:
    comments = ["좋은 댓글", "오늘 날씨 좋다", "ㅅㅂ 꺼져", "병신 같은 소리"]
    labels = [0, 0, 1, 1]
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4))
    features = vectorizer.fit_transform(comments)
    model = LogisticRegression(solver="liblinear", random_state=42)
    model.fit(features, labels)

    artifact_dir.mkdir(parents=True, exist_ok=True)
    save_pickle(artifact_dir / "tfidf_vectorizer.pkl", vectorizer)
    save_pickle(artifact_dir / "logistic_regression.pkl", model)
    (artifact_dir / "model_config.json").write_text(
        json.dumps(
            build_model_config(
                threshold=threshold,
                ngram_min=2,
                ngram_max=4,
                c_value=2.0,
                use_class_weight=True,
            ),
        ),
        encoding="utf-8",
    )
