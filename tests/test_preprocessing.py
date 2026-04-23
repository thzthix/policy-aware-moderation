import pytest

from src.moderation.preprocessing import clean_text, tokenize_text


def test_clean_text_basic_case() -> None:
    assert clean_text("Hello, WORLD!!!") == "hello world"


def test_clean_text_normalizes_whitespace_and_newlines() -> None:
    assert clean_text("Hello\n\n   WORLD\t!") == "hello world"


def test_clean_text_replaces_underscore_with_space() -> None:
    assert clean_text("너_진짜 최악이다") == "너 진짜 최악이다"


def test_clean_text_replaces_underscore_with_space_for_english_text() -> None:
    assert clean_text("bad_comment_123") == "bad comment 123"


def test_clean_text_keeps_korean_characters() -> None:
    assert clean_text("너 진짜 최악이다!!!") == "너 진짜 최악이다"


def test_clean_text_keeps_korean_laughter() -> None:
    assert clean_text("ㅋㅋㅋㅋ 진짜 웃김") == "ㅋㅋㅋㅋ 진짜 웃김"


def test_clean_text_rejects_non_string_input() -> None:
    with pytest.raises(TypeError):
        clean_text(None)


def test_tokenize_text_basic_case() -> None:
    assert tokenize_text("hello toxic world") == ["hello", "toxic", "world"]


def test_tokenize_text_empty_string() -> None:
    assert tokenize_text("") == []


def test_tokenize_text_rejects_non_string_input() -> None:
    with pytest.raises(TypeError):
        tokenize_text(123)
