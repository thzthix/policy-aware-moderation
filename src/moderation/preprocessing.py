import re


def clean_text(text: str) -> str:
    """댓글 텍스트를 기본 정제한다."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    normalized_text = text.replace("\n", " ").lower()
    cleaned_text = re.sub(r"[^a-z0-9\s]", "", normalized_text)
    return " ".join(cleaned_text.split())


def tokenize_text(text: str) -> list[str]:
    """공백 기준으로 댓글을 토큰화한다."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    if not text:
        return []

    return text.split()
