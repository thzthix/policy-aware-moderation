from pathlib import Path

import numpy as np
import pytest
import torch

from src.moderation.predictor import ToxicityPredictor


class MockEmbeddingModel:
    def __init__(self) -> None:
        self.vectors = {
            "hello": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "toxic": np.array([4.0, 5.0, 6.0], dtype=np.float32),
            "OOV": np.array([-1.0, -1.0, -1.0], dtype=np.float32),
        }
        self.key_to_index = {key: index for index, key in enumerate(self.vectors)}
        self.vector_size = 3

    def __getitem__(self, key: str) -> np.ndarray:
        return self.vectors[key]


class MockModel:
    def __init__(self, logit: float) -> None:
        self.logit = logit
        self.received_input: torch.Tensor | None = None
        self.is_eval = False

    def eval(self) -> None:
        self.is_eval = True

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        self.received_input = inputs
        return torch.tensor([[self.logit]], dtype=torch.float32)


def test_from_artifacts_rejects_missing_embedding_model_path(tmp_path: Path) -> None:
    model_path = tmp_path / "model.pth"
    model_path.write_bytes(b"model")

    with pytest.raises(FileNotFoundError):
        ToxicityPredictor.from_artifacts(
            embedding_model_path=tmp_path / "missing.model",
            model_path=model_path,
            input_size=3,
            hidden_size=5,
            num_layers=1,
        )


def test_from_artifacts_rejects_missing_model_path(tmp_path: Path) -> None:
    embedding_path = tmp_path / "word2vec.model"
    embedding_path.write_bytes(b"embedding")

    with pytest.raises(FileNotFoundError):
        ToxicityPredictor.from_artifacts(
            embedding_model_path=embedding_path,
            model_path=tmp_path / "missing.pth",
            input_size=3,
            hidden_size=5,
            num_layers=1,
        )


def test_predict_returns_label_and_score() -> None:
    model = MockModel(logit=2.0)
    predictor = ToxicityPredictor(
        embedding_model=MockEmbeddingModel(),
        model=model,
    )

    result = predictor.predict("Hello toxic!")

    assert set(result) == {"label", "score"}
    assert result["label"] == "toxic"
    assert isinstance(result["score"], float)
    assert 0.0 <= result["score"] <= 1.0
    assert model.is_eval
    assert model.received_input is not None
    assert model.received_input.dtype == torch.float32


def test_predict_uses_threshold_for_toxic_label() -> None:
    predictor = ToxicityPredictor(
        embedding_model=MockEmbeddingModel(),
        model=MockModel(logit=0.0),
        threshold=0.5,
    )

    result = predictor.predict("hello")

    assert result["label"] == "toxic"
    assert result["score"] == pytest.approx(0.5)


def test_predict_uses_threshold_for_non_toxic_label() -> None:
    predictor = ToxicityPredictor(
        embedding_model=MockEmbeddingModel(),
        model=MockModel(logit=-2.0),
        threshold=0.5,
    )

    result = predictor.predict("hello")

    assert result["label"] == "non-toxic"
    assert result["score"] < 0.5
