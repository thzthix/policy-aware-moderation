from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.moderation.tfidf_predictor import TFIDFPredictor


def main() -> None:
    """TF-IDF + LR artifact로 댓글 1건을 예측한다."""
    args = _parse_args()
    predictor = TFIDFPredictor.from_artifacts(args.artifact_dir)
    result = predictor.predict(args.comment)
    print(f"label: {result['label']}")
    print(f"score: {result['score']:.4f}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TF-IDF + Logistic Regression artifact로 댓글 1건을 예측합니다.",
    )
    parser.add_argument("--artifact-dir", required=True, help="artifact 디렉터리")
    parser.add_argument("--comment", required=True, help="예측할 댓글")
    return parser.parse_args()


if __name__ == "__main__":
    main()
