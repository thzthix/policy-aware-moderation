from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.moderation.policy_dataset import load_policy_rules


def test_load_policy_rules_reads_valid_json(tmp_path: Path) -> None:
    json_path = tmp_path / "policy_chunks.json"
    json_path.write_text(
        json.dumps(
            [
                {
                    "policy_id": "attack_001",
                    "category": "insult_or_attack",
                    "concept": "직접적인 모욕",
                    "policy_text": "특정 개인에게 직접적인 모욕을 가하는 표현은 공격적 발화로 본다.",
                    "severity": "medium",
                    "default_action": "review",
                    "source_reference": "meta_bullying_harassment",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    rules = load_policy_rules(json_path)

    assert rules == [
        {
            "policy_id": "attack_001",
            "category": "insult_or_attack",
            "concept": "직접적인 모욕",
            "policy_text": "특정 개인에게 직접적인 모욕을 가하는 표현은 공격적 발화로 본다.",
            "severity": "medium",
            "default_action": "review",
            "source_reference": "meta_bullying_harassment",
        }
    ]


def test_load_policy_rules_raises_for_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_policy_rules(tmp_path / "missing.json")


def test_load_policy_rules_raises_when_top_level_is_not_list(tmp_path: Path) -> None:
    json_path = tmp_path / "policy_chunks.json"
    json_path.write_text(json.dumps({"policy_id": "attack_001"}), encoding="utf-8")

    with pytest.raises(ValueError, match="최상위 구조는 리스트"):
        load_policy_rules(json_path)


def test_load_policy_rules_raises_for_missing_required_field(tmp_path: Path) -> None:
    json_path = tmp_path / "policy_chunks.json"
    json_path.write_text(
        json.dumps(
            [
                {
                    "policy_id": "attack_001",
                    "category": "insult_or_attack",
                    "concept": "직접적인 모욕",
                    "severity": "medium",
                    "default_action": "review",
                    "source_reference": "meta_bullying_harassment",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="policy_text"):
        load_policy_rules(json_path)


def test_load_policy_rules_raises_for_empty_string_field(tmp_path: Path) -> None:
    json_path = tmp_path / "policy_chunks.json"
    json_path.write_text(
        json.dumps(
            [
                {
                    "policy_id": "attack_001",
                    "category": "insult_or_attack",
                    "concept": "직접적인 모욕",
                    "policy_text": " ",
                    "severity": "medium",
                    "default_action": "review",
                    "source_reference": "meta_bullying_harassment",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="policy_text"):
        load_policy_rules(json_path)


def test_load_policy_rules_raises_for_invalid_severity(tmp_path: Path) -> None:
    json_path = tmp_path / "policy_chunks.json"
    json_path.write_text(
        json.dumps(
            [
                {
                    "policy_id": "attack_001",
                    "category": "insult_or_attack",
                    "concept": "직접적인 모욕",
                    "policy_text": "특정 개인에게 직접적인 모욕을 가하는 표현은 공격적 발화로 본다.",
                    "severity": "critical",
                    "default_action": "review",
                    "source_reference": "meta_bullying_harassment",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="severity"):
        load_policy_rules(json_path)


def test_load_policy_rules_raises_for_invalid_default_action(tmp_path: Path) -> None:
    json_path = tmp_path / "policy_chunks.json"
    json_path.write_text(
        json.dumps(
            [
                {
                    "policy_id": "attack_001",
                    "category": "insult_or_attack",
                    "concept": "직접적인 모욕",
                    "policy_text": "특정 개인에게 직접적인 모욕을 가하는 표현은 공격적 발화로 본다.",
                    "severity": "medium",
                    "default_action": "block",
                    "source_reference": "meta_bullying_harassment",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="default_action"):
        load_policy_rules(json_path)
