from __future__ import annotations

import pytest

from src.moderation.policy_documents import Document
from src.moderation.policy_similarity import cosine_similarity, retrieve_top_k


def test_cosine_similarity_returns_high_score_for_same_vector() -> None:
    score = cosine_similarity([1.0, 0.0], [1.0, 0.0])

    assert score == pytest.approx(1.0)


def test_cosine_similarity_returns_low_score_for_different_vector() -> None:
    score = cosine_similarity([1.0, 0.0], [0.0, 1.0])

    assert score == pytest.approx(0.0)


def test_retrieve_top_k_returns_requested_count(monkeypatch) -> None:
    documents = [
        Document(
            page_content="특정 개인에게 모욕적 호칭이나 욕설을 직접 사용하는 표현은 공격적 발화로 본다.",
            metadata={
                "policy_id": "attack_001",
                "category": "insult_or_attack",
                "concept": "직접적인 모욕",
                "severity": "medium",
                "default_action": "review",
                "source_reference": "meta_bullying_harassment",
            },
        ),
        Document(
            page_content="특정 개인의 지능, 이해력, 판단력, 능력을 열등하다고 깎아내리는 표현은 능력 비하로 본다.",
            metadata={
                "policy_id": "attack_002",
                "category": "insult_or_attack",
                "concept": "지능/능력 비하",
                "severity": "medium",
                "default_action": "review",
                "source_reference": "meta_bullying_harassment",
            },
        ),
        Document(
            page_content="보호대상 집단이나 그 구성원을 멸칭이나 모욕적 호칭으로 부르는 표현은 혐오성 공격으로 본다.",
            metadata={
                "policy_id": "hate_002",
                "category": "hate_discrimination",
                "concept": "보호집단 대상 모욕",
                "severity": "high",
                "default_action": "hide",
                "source_reference": "meta_hate_speech",
            },
        ),
    ]
    embeddings = [
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.5],
    ]

    monkeypatch.setattr(
        "src.moderation.policy_similarity.embed_comment",
        lambda text: [1.0, 0.0],
    )

    results = retrieve_top_k(
        "뭐래 ㅂㅅ이",
        documents,
        embeddings,
        category="insult_or_attack",
        top_k=2,
    )

    assert len(results) == 2
    assert results[0]["policy_id"] == "attack_001"
    assert all(result["category"] == "insult_or_attack" for result in results)


def test_retrieve_top_k_returns_one_result_when_top_k_is_one(monkeypatch) -> None:
    documents = [
        Document(
            page_content="특정 개인의 전화번호나 연락처를 본인 동의 없이 공개하거나 공유를 유도하는 표현은 개인정보 노출로 본다.",
            metadata={
                "policy_id": "privacy_002",
                "category": "privacy_or_doxxing",
                "concept": "전화번호 공개",
                "severity": "high",
                "default_action": "hide",
                "source_reference": "meta_privacy_violation",
            },
        ),
        Document(
            page_content="특정 개인의 집 주소나 거주지 식별 정보를 공개하거나 퍼뜨리는 표현은 신상 노출로 본다.",
            metadata={
                "policy_id": "privacy_001",
                "category": "privacy_or_doxxing",
                "concept": "주소 공개",
                "severity": "high",
                "default_action": "hide",
                "source_reference": "meta_privacy_violation",
            },
        ),
    ]
    embeddings = [
        [0.0, 1.0],
        [1.0, 0.0],
    ]

    monkeypatch.setattr(
        "src.moderation.policy_similarity.embed_comment",
        lambda text: [0.0, 1.0],
    )

    results = retrieve_top_k(
        "전화번호: 010-2332-2322",
        documents,
        embeddings,
        category="privacy_or_doxxing",
        top_k=1,
    )

    assert len(results) == 1
    assert results[0]["policy_id"] == "privacy_002"


def test_retrieve_top_k_returns_empty_list_for_unknown_category(monkeypatch) -> None:
    documents = [
        Document(
            page_content="특정 개인에게 모욕적 호칭이나 욕설을 직접 사용하는 표현은 공격적 발화로 본다.",
            metadata={
                "policy_id": "attack_001",
                "category": "insult_or_attack",
                "concept": "직접적인 모욕",
                "severity": "medium",
                "default_action": "review",
                "source_reference": "meta_bullying_harassment",
            },
        )
    ]
    embeddings = [[1.0, 0.0]]

    monkeypatch.setattr(
        "src.moderation.policy_similarity.embed_comment",
        lambda text: [1.0, 0.0],
    )

    results = retrieve_top_k(
        "뭐래 ㅂㅅ이",
        documents,
        embeddings,
        category="privacy_or_doxxing",
        top_k=1,
    )

    assert results == []


def test_retrieve_top_k_returns_top_k_results_when_min_score_is_none(
    monkeypatch,
) -> None:
    documents = [
        Document(
            page_content="특정 개인에게 모욕적 호칭이나 욕설을 직접 사용하는 표현은 공격적 발화로 본다.",
            metadata={
                "policy_id": "attack_001",
                "category": "insult_or_attack",
                "concept": "직접적인 모욕",
                "severity": "medium",
                "default_action": "review",
                "source_reference": "meta_bullying_harassment",
            },
        ),
        Document(
            page_content="특정 개인의 지능, 이해력, 판단력, 능력을 열등하다고 깎아내리는 표현은 능력 비하로 본다.",
            metadata={
                "policy_id": "attack_002",
                "category": "insult_or_attack",
                "concept": "지능/능력 비하",
                "severity": "medium",
                "default_action": "review",
                "source_reference": "meta_bullying_harassment",
            },
        ),
    ]
    embeddings = [
        [1.0, 0.0],
        [0.8, 0.2],
    ]

    monkeypatch.setattr(
        "src.moderation.policy_similarity.embed_comment",
        lambda text: [1.0, 0.0],
    )

    results = retrieve_top_k(
        "뭐래 ㅂㅅ이",
        documents,
        embeddings,
        category="insult_or_attack",
        top_k=2,
        min_score=None,
    )

    assert len(results) == 2


def test_retrieve_top_k_filters_results_by_min_score(monkeypatch) -> None:
    documents = [
        Document(
            page_content="특정 개인에게 모욕적 호칭이나 욕설을 직접 사용하는 표현은 공격적 발화로 본다.",
            metadata={
                "policy_id": "attack_001",
                "category": "insult_or_attack",
                "concept": "직접적인 모욕",
                "severity": "medium",
                "default_action": "review",
                "source_reference": "meta_bullying_harassment",
            },
        ),
        Document(
            page_content="특정 개인의 지능, 이해력, 판단력, 능력을 열등하다고 깎아내리는 표현은 능력 비하로 본다.",
            metadata={
                "policy_id": "attack_002",
                "category": "insult_or_attack",
                "concept": "지능/능력 비하",
                "severity": "medium",
                "default_action": "review",
                "source_reference": "meta_bullying_harassment",
            },
        ),
    ]
    embeddings = [
        [1.0, 0.0],
        [0.8, 0.2],
    ]

    monkeypatch.setattr(
        "src.moderation.policy_similarity.embed_comment",
        lambda text: [1.0, 0.0],
    )

    results = retrieve_top_k(
        "뭐래 ㅂㅅ이",
        documents,
        embeddings,
        category="insult_or_attack",
        top_k=2,
        min_score=0.99,
    )

    assert len(results) == 1
    assert results[0]["policy_id"] == "attack_001"


def test_retrieve_top_k_returns_empty_list_when_all_scores_are_below_min_score(
    monkeypatch,
) -> None:
    documents = [
        Document(
            page_content="특정 개인에게 모욕적 호칭이나 욕설을 직접 사용하는 표현은 공격적 발화로 본다.",
            metadata={
                "policy_id": "attack_001",
                "category": "insult_or_attack",
                "concept": "직접적인 모욕",
                "severity": "medium",
                "default_action": "review",
                "source_reference": "meta_bullying_harassment",
            },
        )
    ]
    embeddings = [[0.8, 0.2]]

    monkeypatch.setattr(
        "src.moderation.policy_similarity.embed_comment",
        lambda text: [1.0, 0.0],
    )

    results = retrieve_top_k(
        "뭐래 ㅂㅅ이",
        documents,
        embeddings,
        category="insult_or_attack",
        top_k=1,
        min_score=0.99,
    )

    assert results == []
