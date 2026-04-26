from __future__ import annotations

from src.moderation.decision_schema import (
    ALLOWED_ACTIONS,
    CATEGORY_LABELS,
    ModerationDecisionResponse,
)


def test_all_category_keys_have_labels() -> None:
    expected_categories = {
        "normal",
        "spam_ad",
        "hate_discrimination",
        "insult_or_attack",
        "toxic_uncertain",
        "uncertain",
    }

    assert set(CATEGORY_LABELS) == expected_categories
    assert all(label for label in CATEGORY_LABELS.values())


def test_all_action_values_are_allowed() -> None:
    assert set(ALLOWED_ACTIONS) == {"allow", "review", "hide"}


def test_example_response_contains_expected_fields() -> None:
    response: ModerationDecisionResponse = {
        "label": "toxic",
        "score": 0.86,
        "category": "insult_or_attack",
        "category_label": "모욕/공격적 표현",
        "recommended_action": "review",
        "reason": "상대방을 직접 비하하는 표현이 포함되어 검토가 필요합니다.",
        "matched_policies": [
            {
                "policy_id": "policy_insult_attack",
                "title": "모욕/공격적 표현",
            },
        ],
        "signals": {
            "has_url": False,
            "has_ad_keyword": False,
            "has_profanity": True,
            "has_hate_expression": False,
        },
    }

    assert set(response) == {
        "label",
        "score",
        "category",
        "category_label",
        "recommended_action",
        "reason",
        "matched_policies",
        "signals",
    }
    assert set(response["signals"]) == {
        "has_url",
        "has_ad_keyword",
        "has_profanity",
        "has_hate_expression",
    }


def test_response_allows_empty_matched_policies() -> None:
    response: ModerationDecisionResponse = {
        "label": "non-toxic",
        "score": 0.14,
        "category": "normal",
        "category_label": "정상",
        "recommended_action": "allow",
        "reason": "명확한 정책 위반 신호가 부족합니다.",
        "matched_policies": [],
        "signals": {
            "has_url": False,
            "has_ad_keyword": False,
            "has_profanity": False,
            "has_hate_expression": False,
        },
    }

    assert response["matched_policies"] == []
