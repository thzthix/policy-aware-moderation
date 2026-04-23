from __future__ import annotations

from pathlib import Path

import torch

from src.moderation.embedding import pad_sequence_vectors, tokens_to_sequence_vectors
from src.moderation.model import GRUModel
from src.moderation.preprocessing import clean_text, tokenize_text


class ToxicityPredictor:
    """댓글 독성 여부를 예측한다."""

    def __init__(self, embedding_model, model, threshold: float = 0.5) -> None:
        self.embedding_model = embedding_model
        self.model = model
        self.threshold = threshold

    @classmethod
    def from_artifacts(
        cls,
        embedding_model_path: str | Path,
        model_path: str | Path,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        threshold: float = 0.5,
    ) -> ToxicityPredictor:
        """저장된 artifact로 predictor를 생성한다."""
        embedding_path = Path(embedding_model_path)
        state_dict_path = Path(model_path)

        if not embedding_path.exists():
            raise FileNotFoundError("embedding model 파일을 찾을 수 없습니다.")

        if not state_dict_path.exists():
            raise FileNotFoundError("model 파일을 찾을 수 없습니다.")

        from gensim.models import Word2Vec

        loaded_embedding_model = Word2Vec.load(str(embedding_path)).wv
        model = GRUModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )
        state_dict = torch.load(state_dict_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()

        return cls(
            embedding_model=loaded_embedding_model,
            model=model,
            threshold=threshold,
        )

    def predict(self, comment: str) -> dict[str, float | str]:
        """댓글 1건의 독성 점수와 라벨을 반환한다."""
        cleaned_comment = clean_text(comment)
        tokens = tokenize_text(cleaned_comment)
        sequence_vectors = tokens_to_sequence_vectors(tokens, self.embedding_model)
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
