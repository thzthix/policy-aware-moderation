from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict


class PolicyRule(TypedDict):
    """정책 룰 데이터를 표현한다."""

    policy_id: str
    category: str
    concept: str
    policy_text: str
    severity: str
    default_action: str
    source_reference: str


def load_policy_rules(path: str | Path) -> list[PolicyRule]:
    """정책 룰 JSON 파일을 불러온다."""
    file_path = Path(path)
    raw_data = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(raw_data, list):
        raise ValueError("정책 룰 JSON의 최상위 구조는 리스트여야 합니다.")

    return [_validate_policy_rule(raw_rule) for raw_rule in raw_data]


def _validate_policy_rule(raw_rule: object) -> PolicyRule:
    if not isinstance(raw_rule, dict):
        raise ValueError("정책 룰 항목은 객체여야 합니다.")

    required_fields = (
        "policy_id",
        "category",
        "concept",
        "policy_text",
        "severity",
        "default_action",
        "source_reference",
    )
    validated_rule: dict[str, str] = {}
    for field_name in required_fields:
        field_value = raw_rule.get(field_name)
        if not isinstance(field_value, str) or not field_value.strip():
            raise ValueError(f"{field_name} 필드는 비어 있지 않은 문자열이어야 합니다.")
        validated_rule[field_name] = field_value

    if validated_rule["severity"] not in {"low", "medium", "high"}:
        raise ValueError("severity 값이 올바르지 않습니다.")

    if validated_rule["default_action"] not in {"allow", "review", "hide"}:
        raise ValueError("default_action 값이 올바르지 않습니다.")

    return validated_rule  # type: ignore[return-value]
