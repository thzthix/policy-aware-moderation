from __future__ import annotations

from typing import Literal, TypedDict


CategoryKey = Literal[
    "normal",
    "spam_ad",
    "hate_discrimination",
    "insult_or_attack",
    "toxic_uncertain",
    "uncertain",
]
CategoryLabel = Literal[
    "정상",
    "스팸/광고",
    "혐오/차별",
    "모욕/공격적 표현",
    "독성 의심",
    "판단보류",
]
RecommendedAction = Literal["allow", "review", "hide"]
PredictorLabel = Literal["toxic", "non-toxic"]

CATEGORY_LABELS: dict[CategoryKey, CategoryLabel] = {
    "normal": "정상",
    "spam_ad": "스팸/광고",
    "hate_discrimination": "혐오/차별",
    "insult_or_attack": "모욕/공격적 표현",
    "toxic_uncertain": "독성 의심",
    "uncertain": "판단보류",
}
ALLOWED_ACTIONS: tuple[RecommendedAction, ...] = ("allow", "review", "hide")


class MatchedPolicy(TypedDict):
    """매칭된 정책 정보를 담는다."""

    policy_id: str
    title: str


class DecisionSignals(TypedDict):
    """정책 판단용 신호를 담는다."""

    has_url: bool
    has_ad_keyword: bool
    has_profanity: bool
    has_hate_expression: bool


class ModerationDecisionResponse(TypedDict):
    """최종 정책 판단 응답을 표현한다."""

    label: PredictorLabel
    score: float
    category: CategoryKey
    category_label: CategoryLabel
    recommended_action: RecommendedAction
    reason: str
    matched_policies: list[MatchedPolicy]
    signals: DecisionSignals
