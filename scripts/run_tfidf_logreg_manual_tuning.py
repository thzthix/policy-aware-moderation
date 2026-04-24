from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.evaluate_tfidf_logreg import _load_training_data, evaluate_configuration


EXPERIMENTS = [
    {
        "name": "baseline",
        "ngram_min": 2,
        "ngram_max": 5,
        "c_value": 1.0,
        "use_class_weight": False,
    },
    {
        "name": "balanced",
        "ngram_min": 2,
        "ngram_max": 5,
        "c_value": 1.0,
        "use_class_weight": True,
    },
    {
        "name": "shorter_ngram",
        "ngram_min": 2,
        "ngram_max": 4,
        "c_value": 1.0,
        "use_class_weight": True,
    },
    {
        "name": "stronger_regularization",
        "ngram_min": 2,
        "ngram_max": 5,
        "c_value": 0.5,
        "use_class_weight": True,
    },
    {
        "name": "longer_ngram",
        "ngram_min": 3,
        "ngram_max": 5,
        "c_value": 1.0,
        "use_class_weight": True,
    },
]


def main() -> None:
    """TF-IDF + LR 1차 수동 튜닝 결과를 비교한다."""
    args = _parse_args()
    comments, labels = _load_training_data(Path(args.train_csv))

    results = [
        evaluate_configuration(
            comments=comments,
            labels=labels,
            val_size=args.val_size,
            random_state=args.random_state,
            threshold=args.threshold,
            ngram_min=experiment["ngram_min"],
            ngram_max=experiment["ngram_max"],
            c_value=experiment["c_value"],
            use_class_weight=experiment["use_class_weight"],
        )
        | {"name": experiment["name"]}
        for experiment in EXPERIMENTS
    ]
    _print_results(results)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TF-IDF + LR 1차 수동 튜닝 결과를 비교합니다.",
    )
    parser.add_argument("--train-csv", required=True, help="학습 CSV 파일 경로")
    parser.add_argument("--val-size", type=float, default=0.2, help="validation 비율")
    parser.add_argument("--random-state", type=int, default=42, help="랜덤 시드")
    parser.add_argument("--threshold", type=float, default=0.5, help="비교용 threshold")
    return parser.parse_args()


def _print_results(results: list[dict]) -> None:
    print("[Manual Tuning Comparison]")
    print("name\tngram\tc\tclass_weight\troc_auc\tf1\tprecision\trecall\taccuracy")
    for result in results:
        metrics = result["metrics"]
        ngram_min, ngram_max = result["ngram_range"]
        class_weight = "balanced" if result["use_class_weight"] else "none"
        print(
            f"{result['name']}\t"
            f"({ngram_min},{ngram_max})\t"
            f"{result['c_value']:.2f}\t"
            f"{class_weight}\t"
            f"{metrics['roc_auc']:.4f}\t"
            f"{metrics['f1']:.4f}\t"
            f"{metrics['precision']:.4f}\t"
            f"{metrics['recall']:.4f}\t"
            f"{metrics['accuracy']:.4f}",
        )


if __name__ == "__main__":
    main()
