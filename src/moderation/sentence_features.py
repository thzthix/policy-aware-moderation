from __future__ import annotations

import re

import numpy as np

PROFANITY_TOKENS = {
    "시발",
    "씨발",
    "ㅅㅂ",
    "병신",
    "븅신",
    "ㅂㅅ",
    "지랄",
    "개새끼",
    "새끼",
    "꺼져",
    "미친",
    "좆",
    "존나",
}

LAUGH_PATTERN = re.compile(r"(ㅋ{2,}|ㅎ{2,}|ㅠ{2,})")


def extract_sentence_features(
    cleaned_comment: str,
    tokens: list[str],
    embedding_model,
    oov_token: str = "OOV",
) -> np.ndarray:
    """문장 단위 특징 벡터를 계산한다."""
    _validate_inputs(cleaned_comment, tokens)

    comment_length = float(len(cleaned_comment))
    profanity_count = float(sum(token in PROFANITY_TOKENS for token in tokens))
    laugh_count = float(len(LAUGH_PATTERN.findall(cleaned_comment)))
    oov_ratio = _calculate_oov_ratio(tokens, embedding_model, oov_token)

    return np.asarray(
        [comment_length, profanity_count, laugh_count, oov_ratio],
        dtype=np.float32,
    )


def build_sentence_feature_matrix(
    cleaned_comments: list[str],
    tokenized_comments: list[list[str]],
    embedding_model,
    oov_token: str = "OOV",
) -> np.ndarray:
    """문장 특징 행렬을 생성한다."""
    if len(cleaned_comments) != len(tokenized_comments):
        raise ValueError("cleaned_comments와 tokenized_comments 길이가 다릅니다.")

    feature_rows = [
        extract_sentence_features(
            cleaned_comment=cleaned_comment,
            tokens=tokens,
            embedding_model=embedding_model,
            oov_token=oov_token,
        )
        for cleaned_comment, tokens in zip(cleaned_comments, tokenized_comments)
    ]
    return np.asarray(feature_rows, dtype=np.float32)


def get_sentence_feature_names() -> list[str]:
    """문장 특징 이름 목록을 반환한다."""
    return [
        "comment_length",
        "profanity_count",
        "laugh_count",
        "oov_ratio",
    ]


def _calculate_oov_ratio(tokens: list[str], embedding_model, oov_token: str) -> float:
    if not tokens:
        return 0.0

    known_tokens = {
        token
        for token in tokens
        if token in embedding_model.key_to_index and token != oov_token
    }
    oov_count = len(tokens) - len(known_tokens)
    return float(oov_count / len(tokens))


def _validate_inputs(cleaned_comment: str, tokens: list[str]) -> None:
    if not isinstance(cleaned_comment, str):
        raise TypeError("cleaned_comment는 문자열이어야 합니다.")

    if not isinstance(tokens, list):
        raise TypeError("tokens는 리스트여야 합니다.")
