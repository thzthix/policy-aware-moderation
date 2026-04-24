from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


JIGSAW_TOXIC_COLUMNS = (
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
)
KOREAN_HATE_SPEECH_TOXIC_LABELS = {"offensive", "hate"}


def convert_kmhas_row(row: dict[str, Any]) -> dict[str, str | int]:
    """K-MHaS 행을 표준 학습 형식으로 변환한다."""
    text = _require_field(row, "text", "K-MHaS")
    labels = _require_field(row, "label", "K-MHaS")
    if not isinstance(labels, list):
        raise ValueError("K-MHaS label은 리스트여야 합니다.")

    return {
        "comment": str(text),
        "toxicity": 0 if 8 in labels else 1,
    }


def convert_jigsaw_row(row: dict[str, Any]) -> dict[str, str | int]:
    """Jigsaw 행을 표준 학습 형식으로 변환한다."""
    comment_text = _require_field(row, "comment_text", "Jigsaw")
    toxicity = int(
        any(
            int(_require_field(row, column, "Jigsaw")) == 1
            for column in JIGSAW_TOXIC_COLUMNS
        ),
    )

    return {
        "comment": str(comment_text),
        "toxicity": toxicity,
    }


def convert_korean_hate_speech_row(row: dict[str, Any]) -> dict[str, str | int]:
    """한국어 혐오표현 데이터 행을 표준 학습 형식으로 변환한다."""
    comment_text = _require_field(row, "comments", "Korean Hate Speech")
    hate_label = str(_require_field(row, "hate", "Korean Hate Speech"))
    if hate_label not in {"none", *KOREAN_HATE_SPEECH_TOXIC_LABELS}:
        raise ValueError("Korean Hate Speech hate 라벨이 올바르지 않습니다.")

    return {
        "comment": str(comment_text),
        "toxicity": int(hate_label in KOREAN_HATE_SPEECH_TOXIC_LABELS),
    }


def write_standard_csv(rows: Iterable[dict[str, str | int]], output_path: Path) -> None:
    """표준 학습 CSV를 저장한다."""
    data = pd.DataFrame(rows, columns=["comment", "toxicity"])
    if data.empty:
        raise ValueError("저장할 데이터가 비어 있습니다.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)


def main() -> None:
    """Hugging Face 데이터셋을 표준 학습 CSV로 변환한다."""
    args = _parse_args()
    if args.dataset == "kmhas":
        rows = _load_kmhas_rows(args.kmhas_split)
    elif args.dataset == "korean-hate-speech":
        rows = _load_korean_hate_speech_rows(args.korean_hate_speech_split)
    elif args.dataset == "jigsaw":
        rows = _load_jigsaw_rows(args.jigsaw_split)
    else:
        rows = [
            *_load_kmhas_rows(args.kmhas_split),
            *_load_jigsaw_rows(args.jigsaw_split),
        ]

    write_standard_csv(rows, Path(args.output_csv))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hugging Face 독성 댓글 데이터셋을 comment,toxicity CSV로 변환합니다.",
    )
    parser.add_argument(
        "--dataset",
        choices=["kmhas", "korean-hate-speech", "jigsaw", "combined"],
        required=True,
        help="변환할 데이터셋",
    )
    parser.add_argument("--output-csv", required=True, help="저장할 CSV 파일 경로")
    parser.add_argument("--kmhas-split", default="train", help="K-MHaS split 이름")
    parser.add_argument(
        "--korean-hate-speech-split",
        default="train",
        help="Korean Hate Speech split 이름",
    )
    parser.add_argument("--jigsaw-split", default="train", help="Jigsaw split 이름")
    return parser.parse_args()


def _load_kmhas_rows(split: str) -> list[dict[str, str | int]]:
    dataset = _load_huggingface_split("jeanlee/kmhas_korean_hate_speech", split)
    return [convert_kmhas_row(row) for row in dataset]


def _load_jigsaw_rows(split: str) -> list[dict[str, str | int]]:
    dataset = _load_huggingface_split(
        "thesofakillers/jigsaw-toxic-comment-classification-challenge",
        split,
    )
    return [convert_jigsaw_row(row) for row in dataset]


def _load_korean_hate_speech_rows(split: str) -> list[dict[str, str | int]]:
    dataset = _load_huggingface_split("nayohan/korean-hate-speech", split)
    return [convert_korean_hate_speech_row(row) for row in dataset]


def _load_huggingface_split(dataset_name: str, split: str):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("datasets 패키지가 필요합니다. requirements.txt를 설치해 주세요.") from exc

    try:
        return load_dataset(dataset_name, split=split)
    except KeyError as exc:
        raise ValueError(f"{dataset_name} 데이터셋에 {split} split이 없습니다.") from exc


def _require_field(row: dict[str, Any], field_name: str, dataset_name: str) -> Any:
    if field_name not in row:
        raise ValueError(f"{dataset_name} 데이터에 {field_name} 컬럼이 필요합니다.")

    return row[field_name]


if __name__ == "__main__":
    main()
