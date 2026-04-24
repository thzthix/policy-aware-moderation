from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.moderation.preprocessing import clean_text


def main() -> None:
    """TF-IDF + LR 독성 분류 성능을 평가한다."""
    args = _parse_args()
    _set_random_seed(args.random_state)

    comments, labels = _load_training_data(Path(args.train_csv))
    cleaned_comments = _preprocess_comments(comments)
    train_comments, val_comments, train_labels, val_labels = train_test_split(
        cleaned_comments,
        labels,
        test_size=args.val_size,
        random_state=args.random_state,
        stratify=_get_stratify_labels(labels),
    )

    vectorizer, model = _train_vectorizer_and_model(
        train_comments=train_comments,
        train_labels=train_labels,
        ngram_min=args.ngram_min,
        ngram_max=args.ngram_max,
        c_value=args.c,
        random_state=args.random_state,
        use_class_weight=args.use_class_weight,
    )
    scores = _predict_scores(model, vectorizer, val_comments)
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
        description="char n-gram TF-IDF + Logistic Regression 독성 분류 성능을 평가합니다.",
    )
    parser.add_argument("--train-csv", required=True, help="학습 CSV 파일 경로")
    parser.add_argument("--val-size", type=float, default=0.2, help="validation 비율")
    parser.add_argument("--random-state", type=int, default=42, help="랜덤 시드")
    parser.add_argument("--threshold", type=float, default=0.3, help="예측 threshold")
    parser.add_argument("--ngram-min", type=int, default=2, help="char n-gram 최소 길이")
    parser.add_argument("--ngram-max", type=int, default=5, help="char n-gram 최대 길이")
    parser.add_argument("--c", type=float, default=1.0, help="LogisticRegression 규제 강도")
    parser.add_argument(
        "--use-class-weight",
        action="store_true",
        help='LogisticRegression에 class_weight="balanced"를 적용합니다.',
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        help="비교할 threshold 목록. 지정하면 threshold별 지표를 출력합니다.",
    )
    return parser.parse_args()


def _set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


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


def _preprocess_comments(comments: list[str]) -> list[str]:
    return [clean_text(comment) for comment in comments]


def _train_vectorizer_and_model(
    train_comments: list[str],
    train_labels: np.ndarray,
    ngram_min: int,
    ngram_max: int,
    c_value: float,
    random_state: int,
    use_class_weight: bool,
) -> tuple[TfidfVectorizer, LogisticRegression]:
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(ngram_min, ngram_max),
    )
    train_features = vectorizer.fit_transform(train_comments)
    model = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
        solver="liblinear",
        C=c_value,
        class_weight="balanced" if use_class_weight else None,
    )
    model.fit(train_features, train_labels)
    return vectorizer, model


def evaluate_configuration(
    comments: list[str],
    labels: np.ndarray,
    *,
    val_size: float,
    random_state: int,
    threshold: float,
    ngram_min: int,
    ngram_max: int,
    c_value: float,
    use_class_weight: bool,
) -> dict[str, Any]:
    cleaned_comments = _preprocess_comments(comments)
    train_comments, val_comments, train_labels, val_labels = train_test_split(
        cleaned_comments,
        labels,
        test_size=val_size,
        random_state=random_state,
        stratify=_get_stratify_labels(labels),
    )
    vectorizer, model = _train_vectorizer_and_model(
        train_comments=train_comments,
        train_labels=train_labels,
        ngram_min=ngram_min,
        ngram_max=ngram_max,
        c_value=c_value,
        random_state=random_state,
        use_class_weight=use_class_weight,
    )
    scores = _predict_scores(model, vectorizer, val_comments)
    return {
        "threshold": threshold,
        "ngram_range": (ngram_min, ngram_max),
        "c_value": c_value,
        "use_class_weight": use_class_weight,
        "metrics": _calculate_metrics(
            labels=val_labels,
            scores=scores,
            threshold=threshold,
        ),
    }


def _predict_scores(
    model: LogisticRegression,
    vectorizer: TfidfVectorizer,
    comments: list[str],
) -> np.ndarray:
    features = vectorizer.transform(comments)
    return model.predict_proba(features)[:, 1].astype(np.float32)


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


def _get_stratify_labels(labels: np.ndarray) -> np.ndarray | None:
    unique_labels, counts = np.unique(labels, return_counts=True)
    if len(unique_labels) < 2 or counts.min() < 2:
        return None

    return labels


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
