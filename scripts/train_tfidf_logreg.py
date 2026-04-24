from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.evaluate_tfidf_logreg import _load_training_data, _preprocess_comments, _train_vectorizer_and_model
from src.moderation.tfidf_predictor import build_model_config, save_pickle


def main() -> None:
    """TF-IDF + LR artifact를 학습하고 저장한다."""
    args = _parse_args()
    _set_random_seed(args.random_state)

    comments, labels = _load_training_data(Path(args.train_csv))
    cleaned_comments = _preprocess_comments(comments)
    vectorizer, model = _train_vectorizer_and_model(
        train_comments=cleaned_comments,
        train_labels=labels,
        ngram_min=args.ngram_min,
        ngram_max=args.ngram_max,
        c_value=args.c,
        random_state=args.random_state,
        use_class_weight=args.use_class_weight,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_pickle(output_dir / "tfidf_vectorizer.pkl", vectorizer)
    save_pickle(output_dir / "logistic_regression.pkl", model)
    (output_dir / "model_config.json").write_text(
        json.dumps(
            build_model_config(
                threshold=args.threshold,
                ngram_min=args.ngram_min,
                ngram_max=args.ngram_max,
                c_value=args.c,
                use_class_weight=args.use_class_weight,
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TF-IDF + Logistic Regression artifact를 학습합니다.",
    )
    parser.add_argument("--train-csv", required=True, help="학습 CSV 파일 경로")
    parser.add_argument(
        "--output-dir",
        default="artifacts/tfidf_logreg",
        help="artifact 저장 디렉터리",
    )
    parser.add_argument("--random-state", type=int, default=42, help="랜덤 시드")
    parser.add_argument("--threshold", type=float, default=0.47, help="예측 threshold")
    parser.add_argument("--ngram-min", type=int, default=2, help="char n-gram 최소 길이")
    parser.add_argument("--ngram-max", type=int, default=4, help="char n-gram 최대 길이")
    parser.add_argument("--c", type=float, default=2.0, help="LogisticRegression 규제 강도")
    parser.add_argument(
        "--use-class-weight",
        action="store_true",
        help='LogisticRegression에 class_weight="balanced"를 적용합니다.',
    )
    return parser.parse_args()


def _set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


if __name__ == "__main__":
    main()
