from __future__ import annotations

import numpy as np
import pytest

from src.moderation.sentence_features import (
    build_sentence_feature_matrix,
    extract_sentence_features,
    get_sentence_feature_names,
)


class MockEmbeddingModel:
    def __init__(self) -> None:
        self.key_to_index = {
            "너": 0,
            "최악이다": 1,
            "ㅋㅋㅋㅋ": 2,
            "OOV": 3,
        }


def test_extract_sentence_features_returns_expected_values() -> None:
    features = extract_sentence_features(
        cleaned_comment="너 ㅅㅂ 최악이다 ㅋㅋㅋㅋ",
        tokens=["너", "ㅅㅂ", "최악이다", "ㅋㅋㅋㅋ"],
        embedding_model=MockEmbeddingModel(),
    )

    assert features.dtype == np.float32
    assert features.tolist() == pytest.approx([14.0, 1.0, 1.0, 0.25])


def test_extract_sentence_features_returns_zero_oov_ratio_for_empty_tokens() -> None:
    features = extract_sentence_features(
        cleaned_comment="",
        tokens=[],
        embedding_model=MockEmbeddingModel(),
    )

    assert features.tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0])


def test_extract_sentence_features_rejects_invalid_tokens() -> None:
    with pytest.raises(TypeError):
        extract_sentence_features("댓글", "not-a-list", MockEmbeddingModel())


def test_get_sentence_feature_names_returns_fixed_order() -> None:
    assert get_sentence_feature_names() == [
        "comment_length",
        "profanity_count",
        "laugh_count",
        "oov_ratio",
    ]


def test_build_sentence_feature_matrix_returns_float32_matrix() -> None:
    matrix = build_sentence_feature_matrix(
        cleaned_comments=["댓글", "ㅋㅋ"],
        tokenized_comments=[["댓글"], ["ㅋㅋ"]],
        embedding_model=MockEmbeddingModel(),
    )

    assert matrix.shape == (2, 4)
    assert matrix.dtype == np.float32
