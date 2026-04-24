from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.evaluate_tfidf_logreg import (
    _calculate_metrics,
    _load_training_data,
    _predict_scores,
    _preprocess_comments,
    _train_vectorizer_and_model,
)


def main() -> None:
    """TF-IDF + LR 하이퍼파라미터를 Optuna로 탐색한다."""
    args = _parse_args()

    try:
        import optuna
    except ImportError as error:
        raise ImportError("optuna 패키지가 필요합니다. requirements.txt를 설치해 주세요.") from error

    comments, labels = _load_training_data(Path(args.train_csv))
    cleaned_comments = _preprocess_comments(comments)

    sampler = optuna.samplers.TPESampler(seed=args.random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        lambda trial: _objective(
            trial=trial,
            comments=cleaned_comments,
            labels=labels,
            n_splits=args.n_splits,
            random_state=args.random_state,
        ),
        n_trials=args.n_trials,
    )

    _print_study_result(study.best_trial)
    if args.output_json:
        _save_study_result(study.best_trial, Path(args.output_json))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TF-IDF + LR 하이퍼파라미터를 Optuna + StratifiedKFold로 탐색합니다.",
    )
    parser.add_argument("--train-csv", required=True, help="학습 CSV 파일 경로")
    parser.add_argument("--n-trials", type=int, default=20, help="Optuna trial 수")
    parser.add_argument("--n-splits", type=int, default=5, help="StratifiedKFold 분할 수")
    parser.add_argument("--random-state", type=int, default=42, help="랜덤 시드")
    parser.add_argument("--output-json", help="best trial 결과 저장 경로")
    return parser.parse_args()


def _objective(
    trial,
    comments: list[str],
    labels: np.ndarray,
    n_splits: int,
    random_state: int,
) -> float:
    params = _sample_params(trial)
    fold_metrics = evaluate_configuration_cv(
        comments=comments,
        labels=labels,
        n_splits=n_splits,
        random_state=random_state,
        ngram_min=params["ngram_min"],
        ngram_max=params["ngram_max"],
        c_value=params["c_value"],
        use_class_weight=params["use_class_weight"],
        threshold=params["threshold"],
    )
    trial.set_user_attr("cv_metrics", fold_metrics)
    return float(fold_metrics["f1"])


def _sample_params(trial) -> dict[str, int | float | bool]:
    ngram_min = trial.suggest_int("ngram_min", 2, 3)
    ngram_max = trial.suggest_int("ngram_max", max(ngram_min + 1, 4), 5)
    return {
        "ngram_min": ngram_min,
        "ngram_max": ngram_max,
        "c_value": trial.suggest_float("c_value", 0.5, 2.0, log=True),
        "use_class_weight": trial.suggest_categorical("use_class_weight", [False, True]),
        "threshold": trial.suggest_float("threshold", 0.4, 0.6),
    }


def evaluate_configuration_cv(
    comments: list[str],
    labels: np.ndarray,
    *,
    n_splits: int,
    random_state: int,
    ngram_min: int,
    ngram_max: int,
    c_value: float,
    use_class_weight: bool,
    threshold: float,
) -> dict[str, float]:
    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    fold_metrics = []

    for train_indices, val_indices in splitter.split(comments, labels):
        train_comments = [comments[index] for index in train_indices]
        val_comments = [comments[index] for index in val_indices]
        train_labels = labels[train_indices]
        val_labels = labels[val_indices]

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
        fold_metrics.append(
            _calculate_metrics(
                labels=val_labels,
                scores=scores,
                threshold=threshold,
            ),
        )

    return _average_metrics(fold_metrics)


def _average_metrics(fold_metrics: list[dict[str, float]]) -> dict[str, float]:
    metric_names = fold_metrics[0].keys()
    return {
        metric_name: float(np.mean([metrics[metric_name] for metrics in fold_metrics]))
        for metric_name in metric_names
    }


def _print_study_result(best_trial) -> None:
    print("[Optuna Best Trial]")
    print(f"value: {best_trial.value:.4f}")
    for key, value in best_trial.params.items():
        print(f"{key}: {value}")

    cv_metrics = best_trial.user_attrs["cv_metrics"]
    print()
    print("[Cross Validation Metrics]")
    for key, value in cv_metrics.items():
        print(f"{key}: {value:.4f}")


def _save_study_result(best_trial, output_path: Path) -> None:
    result = {
        "value": float(best_trial.value),
        "params": best_trial.params,
        "cv_metrics": best_trial.user_attrs["cv_metrics"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
