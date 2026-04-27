from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

from src.moderation.policy_dataset import PolicyRule


class PolicyRuleEvalSample(TypedDict):
    """정책 룰 평가 샘플을 표현한다."""

    comment: str
    expected_policy_ids: list[str]
    expected_category: str
    expected_action: str
    reason: str


def load_policy_rule_eval_samples(path: str | Path) -> list[PolicyRuleEvalSample]:
    """정책 룰 평가 샘플 JSON을 불러온다."""
    file_path = Path(path)
    raw_data = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(raw_data, list):
        raise ValueError("평가 샘플 JSON의 최상위 구조는 리스트여야 합니다.")

    return [_validate_policy_rule_eval_sample(raw_sample) for raw_sample in raw_data]


def build_manual_retrieval_case(
    sample: PolicyRuleEvalSample,
    rules: list[PolicyRule],
) -> dict[str, object]:
    """샘플 댓글의 수동 retrieval 실험 정보를 만든다."""
    expected_rule_map = {rule["policy_id"]: rule for rule in rules}
    missing_policy_ids = [
        policy_id
        for policy_id in sample["expected_policy_ids"]
        if policy_id not in expected_rule_map
    ]
    if missing_policy_ids:
        raise ValueError("평가 샘플에 정의되지 않은 policy_id가 포함되어 있습니다.")

    expected_rules = [
        expected_rule_map[policy_id]
        for policy_id in sample["expected_policy_ids"]
    ]
    candidate_rules = [
        rule
        for rule in rules
        if rule["category"] == sample["expected_category"]
    ]
    representative_rules = [
        rule
        for rule in expected_rules
        if rule["category"] == sample["expected_category"]
    ]

    return {
        "comment": sample["comment"],
        "expected_category": sample["expected_category"],
        "expected_action": sample["expected_action"],
        "reason": sample["reason"],
        "candidate_rules": candidate_rules,
        "expected_rules": expected_rules,
        "representative_rules": representative_rules,
    }


def _validate_policy_rule_eval_sample(raw_sample: object) -> PolicyRuleEvalSample:
    if not isinstance(raw_sample, dict):
        raise ValueError("평가 샘플 항목은 객체여야 합니다.")

    string_fields = ("comment", "expected_category", "expected_action", "reason")
    validated_sample: dict[str, object] = {}
    for field_name in string_fields:
        field_value = raw_sample.get(field_name)
        if not isinstance(field_value, str) or not field_value.strip():
            raise ValueError(f"{field_name} 필드는 비어 있지 않은 문자열이어야 합니다.")
        validated_sample[field_name] = field_value

    expected_policy_ids = raw_sample.get("expected_policy_ids")
    if not isinstance(expected_policy_ids, list) or any(
        not isinstance(policy_id, str) or not policy_id.strip()
        for policy_id in expected_policy_ids
    ):
        raise ValueError("expected_policy_ids 필드는 문자열 리스트여야 합니다.")
    validated_sample["expected_policy_ids"] = expected_policy_ids

    if validated_sample["expected_action"] not in {"allow", "review", "hide"}:
        raise ValueError("expected_action 값이 올바르지 않습니다.")

    return validated_sample  # type: ignore[return-value]
