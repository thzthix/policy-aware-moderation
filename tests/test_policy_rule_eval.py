from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.moderation.policy_dataset import load_policy_rules
from src.moderation.policy_rule_eval import (
    build_manual_retrieval_case,
    load_policy_rule_eval_samples,
)


def test_load_policy_rule_eval_samples_reads_fixture() -> None:
    fixture_path = Path("tests/fixtures/policy_rule_eval_samples.json")

    samples = load_policy_rule_eval_samples(fixture_path)

    assert len(samples) == 15
    assert samples[0]["comment"] == "뭐래 ㅂㅅ이"
    assert samples[0]["expected_policy_ids"] == ["attack_001"]


def test_load_policy_rule_eval_samples_raises_for_non_list_json(tmp_path: Path) -> None:
    fixture_path = tmp_path / "samples.json"
    fixture_path.write_text(json.dumps({"comment": "테스트"}), encoding="utf-8")

    with pytest.raises(ValueError, match="최상위 구조는 리스트"):
        load_policy_rule_eval_samples(fixture_path)


def test_load_policy_rule_eval_samples_raises_for_invalid_policy_id_list(
    tmp_path: Path,
) -> None:
    fixture_path = tmp_path / "samples.json"
    fixture_path.write_text(
        json.dumps(
            [
                {
                    "comment": "테스트",
                    "expected_policy_ids": [1],
                    "expected_category": "insult_or_attack",
                    "expected_action": "review",
                    "reason": "테스트",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="expected_policy_ids"):
        load_policy_rule_eval_samples(fixture_path)


def test_build_manual_retrieval_case_returns_expected_rules() -> None:
    rules = load_policy_rules("policy_chunks.json")
    samples = load_policy_rule_eval_samples("tests/fixtures/policy_rule_eval_samples.json")

    case = build_manual_retrieval_case(samples[0], rules)

    assert case["expected_category"] == "insult_or_attack"
    assert len(case["candidate_rules"]) == 4
    assert [rule["policy_id"] for rule in case["expected_rules"]] == ["attack_001"]
    assert [rule["policy_id"] for rule in case["representative_rules"]] == ["attack_001"]


def test_build_manual_retrieval_case_keeps_cross_category_expected_rules() -> None:
    rules = load_policy_rules("policy_chunks.json")
    samples = load_policy_rule_eval_samples("tests/fixtures/policy_rule_eval_samples.json")

    case = build_manual_retrieval_case(samples[5], rules)

    assert case["expected_category"] == "threat_or_violence"
    assert [rule["policy_id"] for rule in case["expected_rules"]] == [
        "hate_002",
        "violence_001",
    ]
    assert [rule["policy_id"] for rule in case["representative_rules"]] == [
        "violence_001"
    ]


def test_build_manual_retrieval_case_raises_for_unknown_policy_id() -> None:
    rules = load_policy_rules("policy_chunks.json")
    sample = {
        "comment": "테스트",
        "expected_policy_ids": ["unknown_rule"],
        "expected_category": "uncertain",
        "expected_action": "review",
        "reason": "테스트",
    }

    with pytest.raises(ValueError, match="policy_id"):
        build_manual_retrieval_case(sample, rules)
