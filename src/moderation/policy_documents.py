from __future__ import annotations

from langchain_core.documents import Document

from src.moderation.policy_dataset import (
    POLICY_RULE_METADATA_FIELDS,
    PolicyRule,
)


def build_policy_documents(rules: list[PolicyRule]) -> list[Document]:
    """정책 룰 목록을 Document 목록으로 변환한다."""
    return [
        Document(
            page_content=rule["policy_text"],
            metadata={
                field_name: rule[field_name]
                for field_name in POLICY_RULE_METADATA_FIELDS
            },
        )
        for rule in rules
    ]
