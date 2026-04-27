from __future__ import annotations

from src.moderation.policy_documents import build_policy_documents


def test_build_policy_documents_sets_page_content_from_policy_text() -> None:
    rules = [
        {
            "policy_id": "attack_001",
            "category": "insult_or_attack",
            "concept": "직접적인 모욕",
            "policy_text": "특정 개인에게 모욕적 호칭이나 욕설을 직접 사용하는 표현은 공격적 발화로 본다.",
            "severity": "medium",
            "default_action": "review",
            "source_reference": "meta_bullying_harassment",
        }
    ]

    documents = build_policy_documents(rules)

    assert documents[0].page_content == rules[0]["policy_text"]


def test_build_policy_documents_sets_required_metadata_fields() -> None:
    rules = [
        {
            "policy_id": "hate_002",
            "category": "hate_discrimination",
            "concept": "보호집단 대상 모욕",
            "policy_text": "보호대상 집단이나 그 구성원을 멸칭이나 모욕적 호칭으로 부르는 표현은 혐오성 공격으로 본다.",
            "severity": "high",
            "default_action": "hide",
            "source_reference": "meta_hate_speech",
        }
    ]

    documents = build_policy_documents(rules)

    assert documents[0].metadata == {
        "policy_id": "hate_002",
        "category": "hate_discrimination",
        "concept": "보호집단 대상 모욕",
        "severity": "high",
        "default_action": "hide",
        "source_reference": "meta_hate_speech",
    }


def test_build_policy_documents_returns_same_count_as_input_rules() -> None:
    rules = [
        {
            "policy_id": "attack_001",
            "category": "insult_or_attack",
            "concept": "직접적인 모욕",
            "policy_text": "특정 개인에게 모욕적 호칭이나 욕설을 직접 사용하는 표현은 공격적 발화로 본다.",
            "severity": "medium",
            "default_action": "review",
            "source_reference": "meta_bullying_harassment",
        },
        {
            "policy_id": "privacy_002",
            "category": "privacy_or_doxxing",
            "concept": "전화번호 공개",
            "policy_text": "특정 개인의 전화번호나 연락처를 본인 동의 없이 공개하거나 공유를 유도하는 표현은 개인정보 노출로 본다.",
            "severity": "high",
            "default_action": "hide",
            "source_reference": "meta_privacy_violation",
        },
    ]

    documents = build_policy_documents(rules)

    assert len(documents) == len(rules)
