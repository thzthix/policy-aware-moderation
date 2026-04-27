from __future__ import annotations

from src.moderation.policy_documents import Document
from src.moderation.policy_embeddings import embed_policy_documents


def test_embed_policy_documents_returns_one_embedding_per_document(
    monkeypatch,
) -> None:
    documents = [
        Document(
            page_content="특정 개인에게 모욕적 호칭이나 욕설을 직접 사용하는 표현은 공격적 발화로 본다.",
            metadata={},
        )
    ]

    def fake_request_embeddings(texts: list[str]) -> list[list[float]]:
        assert texts == [documents[0].page_content]
        return [[0.1, 0.2, 0.3]]

    monkeypatch.setattr(
        "src.moderation.policy_embeddings._request_embeddings",
        fake_request_embeddings,
    )

    embeddings = embed_policy_documents(documents)

    assert embeddings == [[0.1, 0.2, 0.3]]


def test_embed_policy_documents_returns_same_count_as_documents(
    monkeypatch,
) -> None:
    documents = [
        Document(page_content="첫 번째 정책", metadata={}),
        Document(page_content="두 번째 정책", metadata={}),
    ]

    def fake_request_embeddings(texts: list[str]) -> list[list[float]]:
        assert texts == [document.page_content for document in documents]
        return [[0.1, 0.2], [0.3, 0.4]]

    monkeypatch.setattr(
        "src.moderation.policy_embeddings._request_embeddings",
        fake_request_embeddings,
    )

    embeddings = embed_policy_documents(documents)

    assert len(embeddings) == len(documents)
    assert embeddings == [[0.1, 0.2], [0.3, 0.4]]
