from __future__ import annotations

import pytest

from src.moderation.policy_dataset import load_policy_rules
from src.moderation.policy_retrieval import retrieve_policy_rules


def test_retrieve_policy_rules_returns_top_rule_for_direct_insult() -> None:
    rules = load_policy_rules("policy_chunks.json")

    matched_rules = retrieve_policy_rules(
        comment="뭐래 ㅂㅅ이",
        rules=rules,
        category="insult_or_attack",
        top_k=3,
    )

    assert matched_rules[0]["policy_id"] == "attack_001"


def test_retrieve_policy_rules_returns_top_rule_for_ability_insult() -> None:
    rules = load_policy_rules("policy_chunks.json")

    matched_rules = retrieve_policy_rules(
        comment="뇌에 우동사리 들었냐",
        rules=rules,
        category="insult_or_attack",
        top_k=3,
    )

    assert matched_rules[0]["policy_id"] == "attack_002"


def test_retrieve_policy_rules_returns_top_rule_for_phone_number_exposure() -> None:
    rules = load_policy_rules("policy_chunks.json")

    matched_rules = retrieve_policy_rules(
        comment="전화번호: 010-2332-2322",
        rules=rules,
        category="privacy_or_doxxing",
        top_k=3,
    )

    assert matched_rules[0]["policy_id"] == "privacy_002"


def test_retrieve_policy_rules_returns_empty_when_no_rule_matches() -> None:
    rules = load_policy_rules("policy_chunks.json")

    matched_rules = retrieve_policy_rules(
        comment="오늘 날씨 좋다",
        rules=rules,
        category="insult_or_attack",
        top_k=3,
    )

    assert matched_rules == []


def test_retrieve_policy_rules_rejects_invalid_comment_type() -> None:
    rules = load_policy_rules("policy_chunks.json")

    with pytest.raises(TypeError, match="문자열"):
        retrieve_policy_rules(123, rules, "insult_or_attack")  # type: ignore[arg-type]


def test_retrieve_policy_rules_rejects_invalid_top_k() -> None:
    rules = load_policy_rules("policy_chunks.json")

    with pytest.raises(ValueError, match="top_k"):
        retrieve_policy_rules("뭐래 ㅂㅅ이", rules, "insult_or_attack", top_k=0)
