from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.prepare_toxicity_dataset import (
    convert_jigsaw_row,
    convert_kmhas_row,
    write_standard_csv,
)


def test_convert_kmhas_row_maps_not_hate_label_to_non_toxic() -> None:
    row = {"text": "오늘 날씨 좋다", "label": [8]}

    assert convert_kmhas_row(row) == {
        "comment": "오늘 날씨 좋다",
        "toxicity": 0,
    }


def test_convert_kmhas_row_maps_hate_labels_to_toxic() -> None:
    row = {"text": "악플질 고만해라", "label": [2, 4]}

    assert convert_kmhas_row(row) == {
        "comment": "악플질 고만해라",
        "toxicity": 1,
    }


def test_convert_kmhas_row_rejects_missing_label() -> None:
    with pytest.raises(ValueError, match="label"):
        convert_kmhas_row({"text": "댓글"})


def test_convert_jigsaw_row_maps_clean_comment_to_non_toxic() -> None:
    row = _build_jigsaw_row(comment_text="Thanks for the explanation.")

    assert convert_jigsaw_row(row) == {
        "comment": "Thanks for the explanation.",
        "toxicity": 0,
    }


def test_convert_jigsaw_row_maps_any_toxic_label_to_toxic() -> None:
    row = _build_jigsaw_row(comment_text="Bad comment", insult=1)

    assert convert_jigsaw_row(row) == {
        "comment": "Bad comment",
        "toxicity": 1,
    }


def test_convert_jigsaw_row_rejects_missing_toxic_column() -> None:
    row = _build_jigsaw_row()
    row.pop("threat")

    with pytest.raises(ValueError, match="threat"):
        convert_jigsaw_row(row)


def test_write_standard_csv_saves_comment_and_toxicity_columns(tmp_path: Path) -> None:
    output_path = tmp_path / "train.csv"

    write_standard_csv(
        [
            {"comment": "좋은 댓글", "toxicity": 0},
            {"comment": "Bad comment", "toxicity": 1},
        ],
        output_path,
    )

    data = pd.read_csv(output_path)
    assert data.to_dict("records") == [
        {"comment": "좋은 댓글", "toxicity": 0},
        {"comment": "Bad comment", "toxicity": 1},
    ]


def test_write_standard_csv_rejects_empty_rows(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="비어 있습니다"):
        write_standard_csv([], tmp_path / "train.csv")


def _build_jigsaw_row(
    comment_text: str = "Clean comment",
    toxic: int = 0,
    severe_toxic: int = 0,
    obscene: int = 0,
    threat: int = 0,
    insult: int = 0,
    identity_hate: int = 0,
) -> dict[str, str | int]:
    return {
        "comment_text": comment_text,
        "toxic": toxic,
        "severe_toxic": severe_toxic,
        "obscene": obscene,
        "threat": threat,
        "insult": insult,
        "identity_hate": identity_hate,
    }
