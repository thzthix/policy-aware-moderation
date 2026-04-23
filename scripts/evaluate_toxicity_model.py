from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from gensim.models import Word2Vec
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.moderation.embedding import pad_sequence_vectors, tokens_to_sequence_vectors
from src.moderation.model import GRUModel
from src.moderation.preprocessing import clean_text, tokenize_text


def main() -> None:
    """validation 기준 독성 분류 성능을 평가한다."""
    args = _parse_args()
    _set_random_seed(args.random_state)

    comments, labels = _load_training_data(Path(args.train_csv))
    tokenized_comments = _preprocess_comments(comments)
    word2vec_model = _train_word2vec(
        tokenized_comments=tokenized_comments,
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        workers=args.workers,
        seed=args.random_state,
    )
    sequences = _build_sequences(tokenized_comments, word2vec_model.wv)

    train_sequences, val_sequences, train_labels, val_labels = train_test_split(
        sequences,
        labels,
        test_size=args.val_size,
        random_state=args.random_state,
        stratify=_get_stratify_labels(labels),
    )

    model = GRUModel(
        input_size=word2vec_model.vector_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
    )
    _train_gru(
        model=model,
        train_sequences=train_sequences,
        train_labels=train_labels,
        vector_size=word2vec_model.vector_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
    scores = _predict_scores(
        model=model,
        sequences=val_sequences,
        vector_size=word2vec_model.vector_size,
        batch_size=args.batch_size,
    )
    metrics = _calculate_metrics(
        labels=val_labels,
        scores=scores,
        threshold=args.threshold,
    )
    _print_metrics(metrics)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Word2Vec + GRU 독성 댓글 분류 모델을 validation 기준으로 평가합니다.",
    )
    parser.add_argument("--train-csv", required=True, help="학습 CSV 파일 경로")
    parser.add_argument("--epochs", type=int, default=10, help="학습 epoch 수")
    parser.add_argument("--batch-size", type=int, default=32, help="batch 크기")
    parser.add_argument("--hidden-size", type=int, default=100, help="GRU hidden 크기")
    parser.add_argument("--num-layers", type=int, default=1, help="GRU layer 수")
    parser.add_argument("--lr", type=float, default=0.001, help="Adam learning rate")
    parser.add_argument("--vector-size", type=int, default=100, help="Word2Vec 벡터 크기")
    parser.add_argument("--window", type=int, default=5, help="Word2Vec window 크기")
    parser.add_argument("--min-count", type=int, default=1, help="Word2Vec min_count")
    parser.add_argument("--workers", type=int, default=4, help="Word2Vec worker 수")
    parser.add_argument("--val-size", type=float, default=0.1, help="validation 비율")
    parser.add_argument("--random-state", type=int, default=42, help="랜덤 시드")
    parser.add_argument("--threshold", type=float, default=0.5, help="예측 threshold")
    return parser.parse_args()


def _set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_training_data(train_path: Path) -> tuple[list[str], np.ndarray]:
    if not train_path.exists():
        raise FileNotFoundError("train.csv 파일을 찾을 수 없습니다.")

    data = pd.read_csv(train_path)
    required_columns = {"comment", "toxicity"}
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError("train.csv에 comment와 toxicity 컬럼이 필요합니다.")

    comments = data["comment"].fillna("").astype(str).tolist()
    labels = data["toxicity"].astype(np.float32).to_numpy()
    return comments, labels


def _preprocess_comments(comments: list[str]) -> list[list[str]]:
    return [tokenize_text(clean_text(comment)) for comment in comments]


def _train_word2vec(
    tokenized_comments: list[list[str]],
    vector_size: int,
    window: int,
    min_count: int,
    workers: int,
    seed: int,
) -> Word2Vec:
    sentences = [tokens if tokens else ["OOV"] for tokens in tokenized_comments]
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        seed=seed,
    )
    _ensure_oov_vector(model)
    return model


def _ensure_oov_vector(model: Word2Vec) -> None:
    if "OOV" in model.wv.key_to_index:
        return

    average_vector = np.mean(model.wv.vectors, axis=0).astype(np.float32)
    model.wv.add_vector("OOV", average_vector)


def _build_sequences(tokenized_comments: list[list[str]], embedding_model) -> list[np.ndarray]:
    return [
        tokens_to_sequence_vectors(tokens, embedding_model)
        for tokens in tokenized_comments
    ]


def _get_stratify_labels(labels: np.ndarray) -> np.ndarray | None:
    unique_labels, counts = np.unique(labels, return_counts=True)
    if len(unique_labels) < 2 or counts.min() < 2:
        return None

    return labels


def _train_gru(
    model: GRUModel,
    train_sequences: list[np.ndarray],
    train_labels: np.ndarray,
    vector_size: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> None:
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for _ in range(epochs):
        model.train()
        for batch_sequences, batch_labels in _iter_batches(
            train_sequences,
            train_labels,
            batch_size,
        ):
            inputs = _to_input_tensor(batch_sequences, vector_size)
            targets = _to_label_tensor(batch_labels)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()


def _predict_scores(
    model: GRUModel,
    sequences: list[np.ndarray],
    vector_size: int,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    scores = []

    with torch.no_grad():
        for batch_sequences, _ in _iter_batches(
            sequences,
            np.zeros(len(sequences), dtype=np.float32),
            batch_size,
        ):
            inputs = _to_input_tensor(batch_sequences, vector_size)
            logits = model(inputs)
            batch_scores = torch.sigmoid(logits).reshape(-1).detach().cpu().numpy()
            scores.extend(batch_scores.tolist())

    return np.asarray(scores, dtype=np.float32)


def _calculate_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    predictions = (scores >= threshold).astype(np.float32)
    return {
        "roc_auc": _safe_roc_auc(labels, scores),
        "f1": f1_score(labels, predictions, zero_division=0),
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
    }


def _safe_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")

    return roc_auc_score(labels, scores)


def _iter_batches(
    sequences: list[np.ndarray],
    labels: np.ndarray,
    batch_size: int,
) -> list[tuple[list[np.ndarray], np.ndarray]]:
    return [
        (sequences[index:index + batch_size], labels[index:index + batch_size])
        for index in range(0, len(sequences), batch_size)
    ]


def _to_input_tensor(sequences: list[np.ndarray], vector_size: int) -> torch.Tensor:
    batch_vectors = pad_sequence_vectors(sequences, vector_size=vector_size)
    return torch.tensor(batch_vectors, dtype=torch.float32)


def _to_label_tensor(labels: np.ndarray) -> torch.Tensor:
    return torch.tensor(labels, dtype=torch.float32).reshape(-1, 1)


def _print_metrics(metrics: dict[str, float]) -> None:
    print("[Evaluation Result]")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")


if __name__ == "__main__":
    main()
