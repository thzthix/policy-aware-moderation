import pytest

from src.moderation.preprocessing import clean_text, tokenize_text


def test_clean_text_basic_case() -> None:
    assert clean_text("Hello, WORLD!!!") == "hello world"


def test_clean_text_normalizes_whitespace_and_newlines() -> None:
    assert clean_text("Hello\n\n   WORLD\t!") == "hello world"


def test_clean_text_removes_underscores() -> None:
    assert clean_text("bad_comment_123") == "badcomment123"


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
