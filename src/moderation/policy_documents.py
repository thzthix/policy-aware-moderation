from __future__ import annotations

from dataclasses import dataclass, field

from src.moderation.policy_dataset import PolicyRule

try:
    from langchain_core.documents import Document
except ModuleNotFoundError:
    try:
        from langchain.schema import Document
    except ModuleNotFoundError:
        @dataclass
        class Document:  # type: ignore[no-redef]
            page_content: str
            metadata: dict[str, str] = field(default_factory=dict)


def build_policy_documents(rules: list[PolicyRule]) -> list[Document]:
    """정책 룰 목록을 Document 목록으로 변환한다."""
    return [
        Document(
            page_content=rule["policy_text"],
            metadata={
                "policy_id": rule["policy_id"],
                "category": rule["category"],
                "concept": rule["concept"],
                "severity": rule["severity"],
                "default_action": rule["default_action"],
                "source_reference": rule["source_reference"],
            },
        )
        for rule in rules
    ]
