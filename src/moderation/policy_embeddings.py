from __future__ import annotations

from src.moderation.policy_documents import Document

EMBEDDING_MODEL = "text-embedding-3-small"


def embed_policy_documents(documents: list[Document]) -> list[list[float]]:
    """정책 Document 목록을 임베딩으로 변환한다."""
    page_contents = [document.page_content for document in documents]
    if not page_contents:
        return []

    return _request_embeddings(page_contents)


def _request_embeddings(texts: list[str]) -> list[list[float]]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("openai 패키지가 필요합니다. requirements.txt를 설치해 주세요.") from exc

    client = OpenAI()
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    sorted_data = sorted(response.data, key=lambda item: item.index)
    return [list(item.embedding) for item in sorted_data]
