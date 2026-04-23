from __future__ import annotations

import json
from pathlib import Path

import torch

from src.moderation.embedding import pad_sequence_vectors, tokens_to_sequence_vectors
from src.moderation.model import GRUModel
from src.moderation.preprocessing import clean_text, tokenize_text


class ToxicityPredictor:
    """댓글 독성 여부를 예측한다."""

    def __init__(
        self,
        embedding_model,
        model,
        threshold: float = 0.5,
        oov_token: str = "OOV",
    ) -> None:
        self.embedding_model = embedding_model
        self.model = model
        self.threshold = threshold
        self.oov_token = oov_token

    @classmethod
    def from_artifacts(
        cls,
        artifact_dir: str | Path,
    ) -> ToxicityPredictor:
        """저장된 artifact로 predictor를 생성한다."""
        artifact_path = Path(artifact_dir)
        embedding_path = artifact_path / "word2vec.model"
        state_dict_path = artifact_path / "best_model.pth"
        config_path = artifact_path / "model_config.json"

        from gensim.models import Word2Vec

        _validate_artifact_files(embedding_path, state_dict_path, config_path)
        model_config = _load_model_config(config_path)
        loaded_embedding_model = _load_embedding_model(embedding_path, Word2Vec)
        model = GRUModel(
            input_size=int(model_config["input_size"]),
            hidden_size=int(model_config["hidden_size"]),
            num_layers=int(model_config["num_layers"]),
        )
        _load_model_state(model, state_dict_path)
        model.eval()

        return cls(
            embedding_model=loaded_embedding_model,
            model=model,
            threshold=float(model_config.get("threshold", 0.5)),
            oov_token=str(model_config.get("oov_token", "OOV")),
        )

    def predict(self, comment: str) -> dict[str, float | str]:
        """댓글 1건의 독성 점수와 라벨을 반환한다."""
        cleaned_comment = clean_text(comment)
        tokens = tokenize_text(cleaned_comment)
        sequence_vectors = tokens_to_sequence_vectors(
            tokens,
            self.embedding_model,
            oov_token=self.oov_token,
        )
        batch_vectors = pad_sequence_vectors(
            [sequence_vectors],
            vector_size=self.embedding_model.vector_size,
        )
        model_inputs = torch.tensor(batch_vectors, dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(model_inputs)
            score = torch.sigmoid(logits).flatten()[0].item()

        label = "toxic" if score >= self.threshold else "non-toxic"
        return {"label": label, "score": float(score)}


def _validate_artifact_files(
    embedding_path: Path,
    state_dict_path: Path,
    config_path: Path,
) -> None:
    if not embedding_path.exists():
        raise FileNotFoundError("word2vec.model 파일을 찾을 수 없습니다.")

    if not state_dict_path.exists():
        raise FileNotFoundError("best_model.pth 파일을 찾을 수 없습니다.")

    if not config_path.exists():
        raise FileNotFoundError("model_config.json 파일을 찾을 수 없습니다.")


def _load_model_config(config_path: Path) -> dict[str, int | float | str | bool | None]:
    try:
        model_config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("model_config.json 형식이 올바르지 않습니다.") from error

    required_fields = {"input_size", "hidden_size", "num_layers"}
    missing_fields = required_fields - set(model_config)
    if missing_fields:
        raise ValueError("model_config.json에 필수 모델 설정이 없습니다.")

    # dropout, bidirectional, max_seq_len 등은 향후 모델 확장용 필드입니다.
    return model_config


def _load_embedding_model(embedding_path: Path, word2vec_class):
    try:
        return word2vec_class.load(str(embedding_path)).wv
    except Exception as error:
        raise RuntimeError("word2vec.model 로드에 실패했습니다.") from error


def _load_model_state(model: GRUModel, state_dict_path: Path) -> None:
    try:
        state_dict = torch.load(state_dict_path, map_location="cpu")
        model.load_state_dict(state_dict)
    except Exception as error:
        raise RuntimeError("best_model.pth 로드에 실패했습니다.") from error
