from __future__ import annotations

import json
from pathlib import Path

import torch

from src.moderation.embedding import pad_sequence_vectors, tokens_to_sequence_vectors
from src.moderation.model import GRUModel
from src.moderation.preprocessing import clean_text, tokenize_text
from src.moderation.sentence_features import extract_sentence_features


class ToxicityPredictor:
    """댓글 독성 여부를 예측한다."""

    def __init__(
        self,
        embedding_model,
        model,
        threshold: float = 0.5,
        oov_token: str = "OOV",
        use_sentence_features: bool = False,
    ) -> None:
        self.embedding_model = embedding_model
        self.model = model
        self.threshold = threshold
        self.oov_token = oov_token
        self.use_sentence_features = use_sentence_features

    @classmethod
    def from_artifacts(
        cls,
        artifact_dir: str | Path,
    ) -> ToxicityPredictor:
        """저장된 artifact로 predictor를 생성한다."""
        artifact_path = Path(artifact_dir)
        state_dict_path = artifact_path / "best_model.pth"
        config_path = artifact_path / "model_config.json"

        model_config = _load_model_config(config_path)
        embedding_type = str(model_config.get("embedding_type", "word2vec"))
        embedding_path = artifact_path / _get_embedding_artifact_name(embedding_type)
        _validate_artifact_files(embedding_path, state_dict_path, config_path)
        loaded_embedding_model = _load_embedding_model(embedding_path, embedding_type)
        model = GRUModel(
            input_size=int(model_config["input_size"]),
            hidden_size=int(model_config["hidden_size"]),
            num_layers=int(model_config["num_layers"]),
            feature_size=int(model_config.get("sentence_feature_size", 0)),
        )
        _load_model_state(model, state_dict_path)
        model.eval()

        return cls(
            embedding_model=loaded_embedding_model,
            model=model,
            threshold=float(model_config.get("threshold", 0.5)),
            oov_token=str(model_config.get("oov_token", "OOV")),
            use_sentence_features=bool(model_config.get("use_sentence_features", False)),
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
        sentence_features = _build_sentence_feature_tensor(
            cleaned_comment=cleaned_comment,
            tokens=tokens,
            embedding_model=self.embedding_model,
            oov_token=self.oov_token,
            use_sentence_features=self.use_sentence_features,
        )

        self.model.eval()
        with torch.no_grad():
            logits = self.model(model_inputs, sentence_features=sentence_features)
            score = torch.sigmoid(logits).flatten()[0].item()

        label = "toxic" if score >= self.threshold else "non-toxic"
        return {"label": label, "score": float(score)}


def _validate_artifact_files(
    embedding_path: Path,
    state_dict_path: Path,
    config_path: Path,
) -> None:
    if not embedding_path.exists():
        raise FileNotFoundError(f"{embedding_path.name} 파일을 찾을 수 없습니다.")

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


def _load_embedding_model(embedding_path: Path, embedding_type: str):
    from gensim.models import FastText, Word2Vec

    model_class = _get_embedding_model_class(embedding_type, Word2Vec, FastText)
    try:
        return model_class.load(str(embedding_path)).wv
    except Exception as error:
        raise RuntimeError(f"{embedding_path.name} 로드에 실패했습니다.") from error


def _load_model_state(model: GRUModel, state_dict_path: Path) -> None:
    try:
        state_dict = torch.load(state_dict_path, map_location="cpu")
        model.load_state_dict(state_dict)
    except Exception as error:
        raise RuntimeError("best_model.pth 로드에 실패했습니다.") from error


def _build_sentence_feature_tensor(
    cleaned_comment: str,
    tokens: list[str],
    embedding_model,
    oov_token: str,
    use_sentence_features: bool,
) -> torch.Tensor | None:
    if not use_sentence_features:
        return None

    sentence_features = extract_sentence_features(
        cleaned_comment=cleaned_comment,
        tokens=tokens,
        embedding_model=embedding_model,
        oov_token=oov_token,
    )
    return torch.tensor(sentence_features.reshape(1, -1), dtype=torch.float32)


def _get_embedding_artifact_name(embedding_type: str) -> str:
    if embedding_type == "word2vec":
        return "word2vec.model"

    if embedding_type == "fasttext":
        return "fasttext.model"

    raise ValueError("지원하지 않는 embedding_type입니다.")


def _get_embedding_model_class(embedding_type: str, word2vec_class, fasttext_class):
    if embedding_type == "word2vec":
        return word2vec_class

    if embedding_type == "fasttext":
        return fasttext_class

    raise ValueError("지원하지 않는 embedding_type입니다.")
