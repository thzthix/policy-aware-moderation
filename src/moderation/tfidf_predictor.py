from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from src.moderation.preprocessing import clean_text


class TFIDFPredictor:
    """TF-IDF + LR로 댓글 독성을 예측한다."""

    def __init__(
        self,
        vectorizer,
        model,
        threshold: float = 0.47,
    ) -> None:
        self.vectorizer = vectorizer
        self.model = model
        self.threshold = threshold

    @classmethod
    def from_artifacts(cls, artifact_dir: str | Path) -> TFIDFPredictor:
        """저장된 artifact로 predictor를 생성한다."""
        artifact_path = Path(artifact_dir)
        vectorizer_path = artifact_path / "tfidf_vectorizer.pkl"
        model_path = artifact_path / "logistic_regression.pkl"
        config_path = artifact_path / "model_config.json"

        _validate_artifact_files(vectorizer_path, model_path, config_path)
        model_config = _load_model_config(config_path)
        vectorizer = _load_pickle(vectorizer_path)
        model = _load_pickle(model_path)

        return cls(
            vectorizer=vectorizer,
            model=model,
            threshold=float(model_config.get("threshold", 0.47)),
        )

    def predict(self, comment: str) -> dict[str, float | str]:
        """댓글 1건의 독성 점수와 라벨을 반환한다."""
        cleaned_comment = clean_text(comment)
        features = self.vectorizer.transform([cleaned_comment])
        score = float(self.model.predict_proba(features)[0, 1])
        label = "toxic" if score >= self.threshold else "non-toxic"
        return {
            "label": label,
            "score": score,
        }


def _validate_artifact_files(
    vectorizer_path: Path,
    model_path: Path,
    config_path: Path,
) -> None:
    if not vectorizer_path.exists():
        raise FileNotFoundError("tfidf_vectorizer.pkl 파일을 찾을 수 없습니다.")

    if not model_path.exists():
        raise FileNotFoundError("logistic_regression.pkl 파일을 찾을 수 없습니다.")

    if not config_path.exists():
        raise FileNotFoundError("model_config.json 파일을 찾을 수 없습니다.")


def _load_model_config(config_path: Path) -> dict[str, int | float | str | bool | None]:
    try:
        model_config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("model_config.json 형식이 올바르지 않습니다.") from error

    required_fields = {"model_type", "threshold"}
    missing_fields = required_fields - set(model_config)
    if missing_fields:
        raise ValueError("model_config.json에 필수 설정이 없습니다.")

    return model_config


def _load_pickle(file_path: Path):
    try:
        with file_path.open("rb") as file:
            return pickle.load(file)
    except Exception as error:
        raise RuntimeError(f"{file_path.name} 로드에 실패했습니다.") from error


def build_model_config(
    threshold: float,
    ngram_min: int,
    ngram_max: int,
    c_value: float,
    use_class_weight: bool,
) -> dict[str, int | float | str | bool]:
    """TF-IDF 모델 설정을 생성한다."""
    return {
        "model_type": "tfidf_logreg",
        "threshold": threshold,
        "analyzer": "char",
        "ngram_min": ngram_min,
        "ngram_max": ngram_max,
        "c_value": c_value,
        "use_class_weight": use_class_weight,
    }


def save_pickle(file_path: Path, value) -> None:
    """pickle artifact를 저장한다."""
    with file_path.open("wb") as file:
        pickle.dump(value, file)

