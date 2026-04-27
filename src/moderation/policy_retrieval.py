from __future__ import annotations

from src.moderation.policy_dataset import PolicyRule
from src.moderation.preprocessing import clean_text


RULE_HINTS: dict[str, tuple[str, ...]] = {
    "attack_001": ("병신", "ㅂㅅ", "새끼", "꺼져", "한심"),
    "attack_002": ("지능", "능력", "이해력", "판단력", "머리", "우동사리"),
    "attack_003": ("외모", "몸", "체형", "얼굴", "턱", "돼지"),
    "attack_004": ("ㅋㅋ", "비웃", "비꼬", "빈정", "조롱"),
    "hate_001": ("다", "전부", "원래", "집단"),
    "hate_002": ("김치녀", "한남", "한녀", "틀딱", "장애인", "멸칭"),
    "hate_003": ("벌레", "병균", "괴물", "짐승", "인간 이하"),
    "hate_004": ("배제", "차별", "쫓아내", "권리", "못 오게"),
    "violence_001": ("죽여", "폭행", "패버", "해치", "찾아가"),
    "violence_002": ("때려", "쳐버려", "패", "해치", "가서"),
    "violence_003": ("칼", "총", "폭탄", "방화", "무기"),
    "violence_004": ("맞아도", "죽어야지", "통쾌", "잘했다", "응원"),
    "privacy_001": ("주소", "거주지", "아파트", "집"),
    "privacy_002": ("전화번호", "연락처", "010"),
    "privacy_003": ("학교", "직장", "회사", "근무지", "통학", "출근", "00중"),
    "privacy_004": ("뿌려", "퍼뜨리", "공개", "협박"),
}


def retrieve_policy_rules(
    comment: str,
    rules: list[PolicyRule],
    category: str,
    top_k: int = 3,
) -> list[PolicyRule]:
    """댓글과 가장 가까운 정책 룰 후보를 반환한다."""
    if not isinstance(comment, str):
        raise TypeError("comment는 문자열이어야 합니다.")
    if not isinstance(category, str) or not category.strip():
        raise ValueError("category는 비어 있지 않은 문자열이어야 합니다.")
    if top_k < 1:
        raise ValueError("top_k는 1 이상이어야 합니다.")

    cleaned_comment = clean_text(comment)
    category_rules = [rule for rule in rules if rule["category"] == category]
    scored_rules = [
        (rule, _score_policy_rule(cleaned_comment, rule))
        for rule in category_rules
    ]
    matched_rules = [
        (rule, score)
        for rule, score in scored_rules
        if score > 0
    ]
    matched_rules.sort(
        key=lambda item: (-item[1], item[0]["policy_id"]),
    )
    return [rule for rule, _ in matched_rules[:top_k]]


def _score_policy_rule(cleaned_comment: str, rule: PolicyRule) -> int:
    score = 0
    searchable_text = clean_text(f'{rule["concept"]} {rule["policy_text"]}')
    for token in searchable_text.split():
        if token and token in cleaned_comment:
            score += 1

    for hint in RULE_HINTS.get(rule["policy_id"], ()):
        if hint in cleaned_comment:
            score += 3

    return score
