from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from gensim.models import FastText, Word2Vec
from sklearn.model_selection import train_test_split
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.moderation.embedding import pad_sequence_vectors, tokens_to_sequence_vectors
from src.moderation.model import GRUModel
from src.moderation.preprocessing import clean_text, tokenize_text
from src.moderation.sentence_features import (
    build_sentence_feature_matrix,
    get_sentence_feature_names,
)


def main() -> None:
    """독성 댓글 분류 artifact를 학습하고 저장한다."""
    args = _parse_args()
    _set_random_seed(args.random_state)
    train_path = Path(args.train_csv)
    output_dir = Path(args.output_dir)

    comments, labels = _load_training_data(train_path)
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
    _validate_input_size(args.vector_size, embedding_model.vector_size)
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
    best_state_dict, training_summary = _train_gru(
        model=model,
        train_sequences=train_sequences,
        train_sentence_features=train_sentence_features,
        train_labels=train_labels,
        val_sequences=val_sequences,
        val_sentence_features=val_sentence_features,
        val_labels=val_labels,
        vector_size=embedding_model.vector_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        use_class_weight=args.use_class_weight,
    )

    _save_artifacts(
        output_dir=output_dir,
        embedding_model=embedding_model,
        embedding_type=args.embedding_type,
        model_state_dict=best_state_dict,
        model_config={
            "input_size": embedding_model.vector_size,
            "vector_size": embedding_model.vector_size,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "dropout": 0.0,
            "bidirectional": False,
            "max_seq_len": None,
            "threshold": args.threshold,
            "oov_token": "OOV",
            "embedding_type": args.embedding_type,
            "use_sentence_features": args.use_sentence_features,
            "sentence_feature_size": _get_sentence_feature_size(sentence_features),
            "sentence_feature_names": (
                get_sentence_feature_names() if args.use_sentence_features else []
            ),
            "use_class_weight": args.use_class_weight,
        },
        training_summary={
            **training_summary,
            "train_size": len(train_sequences),
            "valid_size": len(val_sequences),
        },
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Word2Vec + GRU 독성 댓글 분류 artifact를 학습합니다.",
    )
    parser.add_argument("--train-csv", required=True, help="학습 CSV 파일 경로")
    parser.add_argument(
        "--output-dir",
        default="artifacts/classifier",
        help="artifact 저장 디렉터리",
    )
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


def _validate_input_size(expected_size: int, actual_size: int) -> None:
    if expected_size != actual_size:
        raise ValueError("input_size와 Word2Vec vector_size가 일치하지 않습니다.")


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
    val_sequences: list[np.ndarray],
    val_sentence_features: np.ndarray | None,
    val_labels: np.ndarray,
    vector_size: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    use_class_weight: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, int | float]]:
    criterion = _build_loss(train_labels, use_class_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_val_loss = float("inf")
    best_epoch = 0
    last_train_loss = 0.0
    best_state_dict = model.state_dict()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_total = 0.0
        train_count = 0
        for batch_sequences, batch_sentence_features, batch_labels in _iter_batches(
            train_sequences,
            train_labels,
            batch_size,
            sentence_features=train_sentence_features,
        ):
            input_tensor, feature_tensor, targets = _prepare_batch_tensors(
                batch_sequences=batch_sequences,
                batch_labels=batch_labels,
                vector_size=vector_size,
                batch_sentence_features=batch_sentence_features,
            )

            optimizer.zero_grad()
            logits = model(input_tensor, sentence_features=feature_tensor)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            batch_count = len(batch_sequences)
            train_loss_total += loss.item() * batch_count
            train_count += batch_count

        if train_count == 0:
            raise ValueError("train 데이터가 비어 있습니다.")

        last_train_loss = train_loss_total / train_count

        val_loss = _evaluate_loss(
            model,
            val_sequences,
            val_sentence_features,
            val_labels,
            vector_size,
            batch_size,
            criterion,
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    return best_state_dict, {
        "best_epoch": best_epoch,
        "train_loss": last_train_loss,
        "valid_loss": best_val_loss,
    }


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


def _evaluate_loss(
    model: GRUModel,
    sequences: list[np.ndarray],
    sentence_features: np.ndarray | None,
    labels: np.ndarray,
    vector_size: int,
    batch_size: int,
    criterion: nn.Module,
) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for batch_sequences, batch_sentence_features, batch_labels in _iter_batches(
            sequences,
            labels,
            batch_size,
            sentence_features=sentence_features,
        ):
            inputs, feature_tensor, targets = _prepare_batch_tensors(
                batch_sequences=batch_sequences,
                batch_labels=batch_labels,
                vector_size=vector_size,
                batch_sentence_features=batch_sentence_features,
            )
            logits = model(inputs, sentence_features=feature_tensor)
            loss = criterion(logits, targets)
            batch_count = len(batch_sequences)
            total_loss += loss.item() * batch_count
            total_count += batch_count

    if total_count == 0:
        raise ValueError("validation 데이터가 비어 있습니다.")

    return total_loss / total_count


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


def _save_artifacts(
    output_dir: Path,
    embedding_model,
    embedding_type: str,
    model_state_dict: dict[str, torch.Tensor],
    model_config: dict[str, int | float | str | bool | None],
    training_summary: dict[str, int | float],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    embedding_model.save(str(output_dir / _get_embedding_artifact_name(embedding_type)))
    torch.save(model_state_dict, output_dir / "best_model.pth")

    config_path = output_dir / "model_config.json"
    config_path.write_text(
        json.dumps(model_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary_path = output_dir / "training_summary.json"
    summary_path.write_text(
        json.dumps(training_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _get_embedding_artifact_name(embedding_type: str) -> str:
    if embedding_type == "word2vec":
        return "word2vec.model"

    if embedding_type == "fasttext":
        return "fasttext.model"

    raise ValueError("지원하지 않는 embedding_type입니다.")


if __name__ == "__main__":
    main()
