from __future__ import annotations

import math

from src.moderation.policy_documents import Document
from src.moderation.policy_embeddings import _request_embeddings


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """두 벡터의 cosine similarity를 계산한다."""
    if len(vec1) != len(vec2):
        raise ValueError("두 벡터의 길이는 같아야 합니다.")
    if not vec1:
        raise ValueError("벡터는 비어 있을 수 없습니다.")

    dot_product = sum(left * right for left, right in zip(vec1, vec2))
    vec1_norm = math.sqrt(sum(value * value for value in vec1))
    vec2_norm = math.sqrt(sum(value * value for value in vec2))
    if vec1_norm == 0 or vec2_norm == 0:
        raise ValueError("0 벡터는 유사도를 계산할 수 없습니다.")

    return dot_product / (vec1_norm * vec2_norm)


def embed_comment(text: str) -> list[float]:
    """댓글 1건을 임베딩으로 변환한다."""
    if not isinstance(text, str):
        raise TypeError("text는 문자열이어야 합니다.")

    embeddings = _request_embeddings([text])
    return embeddings[0]


def retrieve_top_k(
    comment: str,
    documents: list[Document],
    embeddings: list[list[float]],
    category: str,
    top_k: int = 3,
) -> list[dict[str, object]]:
    """댓글과 가장 유사한 정책 rule 상위 k개를 반환한다."""
    if top_k < 1:
        raise ValueError("top_k는 1 이상이어야 합니다.")
    if len(documents) != len(embeddings):
        raise ValueError("documents와 embeddings의 길이는 같아야 합니다.")

    filtered_pairs = [
        (document, policy_embedding)
        for document, policy_embedding in zip(documents, embeddings)
        if document.metadata["category"] == category
    ]
    if not filtered_pairs:
        return []

    comment_embedding = embed_comment(comment)
    scored_results = []
    for document, policy_embedding in filtered_pairs:
        score = cosine_similarity(comment_embedding, policy_embedding)
        scored_results.append(
            {
                "policy_id": document.metadata["policy_id"],
                "category": document.metadata["category"],
                "concept": document.metadata["concept"],
                "policy_text": document.page_content,
                "severity": document.metadata["severity"],
                "default_action": document.metadata["default_action"],
                "score": score,
            }
        )

    scored_results.sort(key=lambda item: item["score"], reverse=True)
    return scored_results[:top_k]
