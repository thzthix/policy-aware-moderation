from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.moderation.predictor import ToxicityPredictor

SAMPLE_COMMENTS = [
    "오늘 날씨 좋다",
    "너 진짜 최악이다",
    "asdf qwer zxcv",
    "",
]


def main() -> None:
    """저장된 artifact로 샘플 댓글 예측을 확인한다."""
    args = _parse_args()
    try:
        print(f"[INFO] artifact_dir: {args.artifact_dir}")
        predictor = ToxicityPredictor.from_artifacts(args.artifact_dir)
        print("[INFO] predictor loaded successfully")
        print()
        _run_smoke_test(predictor)
    except FileNotFoundError as error:
        print(f"artifact 파일을 찾을 수 없습니다: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    except Exception as error:
        print(f"predictor smoke test 실행에 실패했습니다: {error}", file=sys.stderr)
        raise SystemExit(1) from error


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="저장된 classifier artifact로 predictor smoke test를 실행합니다.",
    )
    parser.add_argument(
        "--artifact-dir",
        required=True,
        help="classifier artifact 디렉터리",
    )
    return parser.parse_args()


def _run_smoke_test(predictor: ToxicityPredictor) -> None:
    for comment in SAMPLE_COMMENTS:
        result = predictor.predict(comment)
        score = result["score"]
        if not 0.0 <= score <= 1.0:
            raise ValueError("score가 0~1 범위를 벗어났습니다.")

        _print_result(comment, result["label"], score)


def _print_result(comment: str, label: str, score: float) -> None:
    print(f"[INPUT] {comment!r}")
    print(f"label: {label}")
    print(f"score: {score:.4f}")
    print()


if __name__ == "__main__":
    main()
