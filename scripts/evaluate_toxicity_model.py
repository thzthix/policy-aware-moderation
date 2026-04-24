from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from gensim.models import FastText, Word2Vec
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.moderation.embedding import pad_sequence_vectors, tokens_to_sequence_vectors
from src.moderation.model import GRUModel
from src.moderation.preprocessing import clean_text, tokenize_text
from src.moderation.sentence_features import build_sentence_feature_matrix


def main() -> None:
    """validation 기준 독성 분류 성능을 평가한다."""
    args = _parse_args()
    _set_random_seed(args.random_state)

    comments, labels = _load_training_data(Path(args.train_csv))
    cleaned_comments, tokenized_comments = _preprocess_comments(comments)
    embedding_model = _train_embedding_model(
        embedding_type=args.embedding_type,
        tokenized_comments=tokenized_comments,
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        workers=args.workers,
        seed=args.random_state,
    )
    sequences = _build_sequences(tokenized_comments, embedding_model.wv)
    sentence_features = _build_sentence_features(
        cleaned_comments=cleaned_comments,
        tokenized_comments=tokenized_comments,
        embedding_model=embedding_model.wv,
        use_sentence_features=args.use_sentence_features,
    )

    (
        train_sequences,
        val_sequences,
        train_sentence_features,
        val_sentence_features,
        train_labels,
        val_labels,
    ) = _split_dataset(
        sequences=sequences,
        sentence_features=sentence_features,
        labels=labels,
        val_size=args.val_size,
        random_state=args.random_state,
    )

    model = GRUModel(
        input_size=embedding_model.vector_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        feature_size=_get_sentence_feature_size(sentence_features),
    )
    _train_gru(
        model=model,
        train_sequences=train_sequences,
        train_sentence_features=train_sentence_features,
        train_labels=train_labels,
        vector_size=embedding_model.vector_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        use_class_weight=args.use_class_weight,
    )
    scores = _predict_scores(
        model=model,
        sequences=val_sequences,
        sentence_features=val_sentence_features,
        vector_size=embedding_model.vector_size,
        batch_size=args.batch_size,
    )
    if args.thresholds:
        threshold_metrics = _calculate_metrics_by_threshold(
            labels=val_labels,
            scores=scores,
            thresholds=args.thresholds,
        )
        _print_threshold_comparison(threshold_metrics)
    else:
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
    parser.add_argument(
        "--embedding-type",
        choices=["word2vec", "fasttext"],
        default="word2vec",
        help="시퀀스 임베딩 학습 방식",
    )
    parser.add_argument("--vector-size", type=int, default=100, help="Word2Vec 벡터 크기")
    parser.add_argument("--window", type=int, default=5, help="Word2Vec window 크기")
    parser.add_argument("--min-count", type=int, default=1, help="Word2Vec min_count")
    parser.add_argument("--workers", type=int, default=4, help="Word2Vec worker 수")
    parser.add_argument("--val-size", type=float, default=0.1, help="validation 비율")
    parser.add_argument("--random-state", type=int, default=42, help="랜덤 시드")
    parser.add_argument("--threshold", type=float, default=0.5, help="예측 threshold")
    parser.add_argument(
        "--use-class-weight",
        action="store_true",
        help="toxic class 비율에 맞춰 BCEWithLogitsLoss pos_weight를 적용합니다.",
    )
    parser.add_argument(
        "--use-sentence-features",
        action="store_true",
        help="GRU 마지막 hidden state 뒤에 문장 단위 feature를 concat합니다.",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        help="비교할 threshold 목록. 지정하면 한 번 학습한 score로 threshold별 지표를 출력합니다.",
    )
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


def _preprocess_comments(comments: list[str]) -> tuple[list[str], list[list[str]]]:
    cleaned_comments = [clean_text(comment) for comment in comments]
    tokenized_comments = [tokenize_text(comment) for comment in cleaned_comments]
    return cleaned_comments, tokenized_comments


def _train_embedding_model(
    embedding_type: str,
    tokenized_comments: list[list[str]],
    vector_size: int,
    window: int,
    min_count: int,
    workers: int,
    seed: int,
):
    if embedding_type == "word2vec":
        return _train_word2vec(
            tokenized_comments=tokenized_comments,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers,
            seed=seed,
        )

    if embedding_type == "fasttext":
        return _train_fasttext(
            tokenized_comments=tokenized_comments,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers,
            seed=seed,
        )

    raise ValueError("지원하지 않는 embedding_type입니다.")


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


def _train_fasttext(
    tokenized_comments: list[list[str]],
    vector_size: int,
    window: int,
    min_count: int,
    workers: int,
    seed: int,
) -> FastText:
    sentences = [tokens if tokens else ["OOV"] for tokens in tokenized_comments]
    model = FastText(
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
    if isinstance(model, FastText):
        return

    if "OOV" in model.wv.key_to_index:
        return

    average_vector = np.mean(model.wv.vectors, axis=0).astype(np.float32)
    model.wv.add_vector("OOV", average_vector)


def _build_sequences(tokenized_comments: list[list[str]], embedding_model) -> list[np.ndarray]:
    return [
        tokens_to_sequence_vectors(tokens, embedding_model)
        for tokens in tokenized_comments
    ]


def _build_sentence_features(
    cleaned_comments: list[str],
    tokenized_comments: list[list[str]],
    embedding_model,
    use_sentence_features: bool,
) -> np.ndarray | None:
    if not use_sentence_features:
        return None

    return build_sentence_feature_matrix(
        cleaned_comments=cleaned_comments,
        tokenized_comments=tokenized_comments,
        embedding_model=embedding_model,
    )


def _split_dataset(
    sequences: list[np.ndarray],
    sentence_features: np.ndarray | None,
    labels: np.ndarray,
    val_size: float,
    random_state: int,
) -> tuple[
    list[np.ndarray],
    list[np.ndarray],
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray,
    np.ndarray,
]:
    indices = np.arange(len(labels))
    train_indices, val_indices = train_test_split(
        indices,
        test_size=val_size,
        random_state=random_state,
        stratify=_get_stratify_labels(labels),
    )
    train_sequences = [sequences[index] for index in train_indices]
    val_sequences = [sequences[index] for index in val_indices]
    train_sentence_features = _slice_sentence_features(sentence_features, train_indices)
    val_sentence_features = _slice_sentence_features(sentence_features, val_indices)
    return (
        train_sequences,
        val_sequences,
        train_sentence_features,
        val_sentence_features,
        labels[train_indices],
        labels[val_indices],
    )


def _get_stratify_labels(labels: np.ndarray) -> np.ndarray | None:
    unique_labels, counts = np.unique(labels, return_counts=True)
    if len(unique_labels) < 2 or counts.min() < 2:
        return None

    return labels


def _train_gru(
    model: GRUModel,
    train_sequences: list[np.ndarray],
    train_sentence_features: np.ndarray | None,
    train_labels: np.ndarray,
    vector_size: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    use_class_weight: bool,
) -> None:
    criterion = _build_loss(train_labels, use_class_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for _ in range(epochs):
        model.train()
        for batch_sequences, batch_sentence_features, batch_labels in _iter_batches(
            train_sequences,
            train_labels,
            batch_size,
            sentence_features=train_sentence_features,
        ):
            inputs, feature_tensor, targets = _prepare_batch_tensors(
                batch_sequences=batch_sequences,
                batch_labels=batch_labels,
                vector_size=vector_size,
                batch_sentence_features=batch_sentence_features,
            )

            optimizer.zero_grad()
            logits = model(inputs, sentence_features=feature_tensor)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()


def _predict_scores(
    model: GRUModel,
    sequences: list[np.ndarray],
    sentence_features: np.ndarray | None,
    vector_size: int,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    scores = []

    with torch.no_grad():
        for batch_sequences, batch_sentence_features, _ in _iter_batches(
            sequences,
            np.zeros(len(sequences), dtype=np.float32),
            batch_size,
            sentence_features=sentence_features,
        ):
            inputs, feature_tensor, _ = _prepare_batch_tensors(
                batch_sequences=batch_sequences,
                batch_labels=np.zeros(len(batch_sequences), dtype=np.float32),
                vector_size=vector_size,
                batch_sentence_features=batch_sentence_features,
            )
            logits = model(inputs, sentence_features=feature_tensor)
            batch_scores = torch.sigmoid(logits).reshape(-1).detach().cpu().numpy()
            scores.extend(batch_scores.tolist())

    return np.asarray(scores, dtype=np.float32)


def _calculate_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    predictions = (scores >= threshold).astype(np.float32)

    # roc_auc: threshold와 관계없이 전체 분류 성능을 본다. 0.5는 랜덤, 1.0은 완벽이다.
    # f1: precision과 recall의 조화 평균이다. 둘 중 하나라도 낮으면 점수가 낮아진다.
    # accuracy: 전체 샘플 중 맞춘 비율이다. 데이터 불균형이 있으면 misleading할 수 있다.
    # precision: toxic이라고 예측한 것 중 실제 toxic 비율이다.
    # recall: 실제 toxic 중 모델이 잡아낸 비율이다.
    return {
        "roc_auc": _safe_roc_auc(labels, scores),
        "f1": f1_score(labels, predictions, zero_division=0),
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
    }


def _calculate_metrics_by_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    thresholds: list[float],
) -> dict[float, dict[str, float]]:
    return {
        threshold: _calculate_metrics(labels, scores, threshold)
        for threshold in thresholds
    }


def _safe_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")

    return roc_auc_score(labels, scores)


def _iter_batches(
    sequences: list[np.ndarray],
    labels: np.ndarray,
    batch_size: int,
    sentence_features: np.ndarray | None = None,
) -> list[tuple[list[np.ndarray], np.ndarray | None, np.ndarray]]:
    return [
        (
            sequences[index:index + batch_size],
            _slice_batch_features(sentence_features, index, index + batch_size),
            labels[index:index + batch_size],
        )
        for index in range(0, len(sequences), batch_size)
    ]


def _to_input_tensor(sequences: list[np.ndarray], vector_size: int) -> torch.Tensor:
    batch_vectors = pad_sequence_vectors(sequences, vector_size=vector_size)
    return torch.tensor(batch_vectors, dtype=torch.float32)


def _to_feature_tensor(sentence_features: np.ndarray | None) -> torch.Tensor | None:
    if sentence_features is None:
        return None

    return torch.tensor(sentence_features, dtype=torch.float32)


def _to_label_tensor(labels: np.ndarray) -> torch.Tensor:
    return torch.tensor(labels, dtype=torch.float32).reshape(-1, 1)


def _prepare_batch_tensors(
    batch_sequences: list[np.ndarray],
    batch_labels: np.ndarray,
    vector_size: int,
    batch_sentence_features: np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    return (
        _to_input_tensor(batch_sequences, vector_size),
        _to_feature_tensor(batch_sentence_features),
        _to_label_tensor(batch_labels),
    )


def _build_loss(labels: np.ndarray, use_class_weight: bool) -> nn.Module:
    if not use_class_weight:
        return nn.BCEWithLogitsLoss()

    pos_weight = _calculate_pos_weight(labels)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def _calculate_pos_weight(labels: np.ndarray) -> torch.Tensor:
    positive_count = float(np.sum(labels == 1))
    negative_count = float(np.sum(labels == 0))
    if positive_count == 0 or negative_count == 0:
        raise ValueError("class weight를 계산하려면 두 클래스가 모두 필요합니다.")

    # BCEWithLogitsLoss의 pos_weight는 positive loss에 곱해져 toxic 클래스를 더 크게 학습시킨다.
    pos_weight_value = negative_count / positive_count
    return torch.tensor([pos_weight_value], dtype=torch.float32)


def _slice_sentence_features(
    sentence_features: np.ndarray | None,
    indices: np.ndarray,
) -> np.ndarray | None:
    if sentence_features is None:
        return None

    return sentence_features[indices]


def _slice_batch_features(
    sentence_features: np.ndarray | None,
    start_index: int,
    end_index: int,
) -> np.ndarray | None:
    if sentence_features is None:
        return None

    return sentence_features[start_index:end_index]


def _get_sentence_feature_size(sentence_features: np.ndarray | None) -> int:
    if sentence_features is None:
        return 0

    return int(sentence_features.shape[1])


def _print_metrics(metrics: dict[str, float]) -> None:
    print("[Evaluation Result]")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")


def _print_threshold_comparison(
    threshold_metrics: dict[float, dict[str, float]],
) -> None:
    print("[Threshold Comparison]")
    for threshold, metrics in threshold_metrics.items():
        print()
        print(f"threshold={threshold:.2f}")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")


if __name__ == "__main__":
    main()
