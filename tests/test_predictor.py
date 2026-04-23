import json
from pathlib import Path

import numpy as np
import pytest
import torch
from gensim.models import Word2Vec

from src.moderation.model import GRUModel
from src.moderation.predictor import ToxicityPredictor

SAMPLE_COMMENTS = [
    "오늘 날씨 좋다",
    "너 진짜 최악이다",
    "asdf qwer zxcv",
    "",
]


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
        self.grad_enabled: bool | None = None

    def eval(self) -> None:
        self.is_eval = True

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        self.received_input = inputs
        self.grad_enabled = torch.is_grad_enabled()
        return torch.tensor([[self.logit]], dtype=torch.float32)


def test_from_artifacts_rejects_missing_embedding_model_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ToxicityPredictor.from_artifacts(tmp_path)


def test_from_artifacts_rejects_missing_model_path(tmp_path: Path) -> None:
    _save_word2vec_artifact(tmp_path)

    with pytest.raises(FileNotFoundError):
        ToxicityPredictor.from_artifacts(tmp_path)


def test_from_artifacts_rejects_missing_model_config(tmp_path: Path) -> None:
    _save_word2vec_artifact(tmp_path)
    torch.save({}, tmp_path / "best_model.pth")

    with pytest.raises(FileNotFoundError):
        ToxicityPredictor.from_artifacts(tmp_path)


def test_from_artifacts_loads_artifacts_and_predicts(tmp_path: Path) -> None:
    _save_word2vec_artifact(tmp_path, oov_token="CUSTOM_OOV")
    model = GRUModel(input_size=3, hidden_size=5, num_layers=1)
    torch.save(model.state_dict(), tmp_path / "best_model.pth")
    _save_model_config(
        tmp_path,
        input_size=3,
        hidden_size=5,
        num_layers=1,
        oov_token="CUSTOM_OOV",
    )

    predictor = ToxicityPredictor.from_artifacts(tmp_path)

    result = predictor.predict("hello unknown")

    assert predictor.oov_token == "CUSTOM_OOV"
    assert set(result) == {"label", "score"}
    assert result["label"] in {"toxic", "non-toxic"}
    assert isinstance(result["score"], float)
    assert 0.0 <= result["score"] <= 1.0


def test_from_artifacts_predicts_sample_comments(tmp_path: Path) -> None:
    _save_word2vec_artifact(tmp_path)
    model = GRUModel(input_size=3, hidden_size=5, num_layers=1)
    torch.save(model.state_dict(), tmp_path / "best_model.pth")
    _save_model_config(
        tmp_path,
        input_size=3,
        hidden_size=5,
        num_layers=1,
    )

    predictor = ToxicityPredictor.from_artifacts(tmp_path)

    for comment in SAMPLE_COMMENTS:
        result = predictor.predict(comment)
        assert set(result) == {"label", "score"}
        assert result["label"] in {"toxic", "non-toxic"}
        assert isinstance(result["score"], float)
        assert 0.0 <= result["score"] <= 1.0


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
    assert model.grad_enabled is False
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


def _save_word2vec_artifact(artifact_dir: Path, oov_token: str = "OOV") -> None:
    word2vec_model = Word2Vec(
        sentences=[["hello", "toxic"], [oov_token, "hello"]],
        vector_size=3,
        min_count=1,
        workers=1,
        seed=42,
    )
    word2vec_model.save(str(artifact_dir / "word2vec.model"))


def _save_model_config(
    artifact_dir: Path,
    input_size: int,
    hidden_size: int,
    num_layers: int,
    oov_token: str = "OOV",
) -> None:
    model_config = {
        "input_size": input_size,
        "vector_size": input_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": 0.0,
        "bidirectional": False,
        "max_seq_len": None,
        "threshold": 0.5,
        "oov_token": oov_token,
    }
    (artifact_dir / "model_config.json").write_text(
        json.dumps(model_config),
        encoding="utf-8",
    )
