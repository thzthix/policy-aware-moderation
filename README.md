# Policy-Aware Moderation

커뮤니티 운영자를 위한 정책 기반 댓글 모더레이션 의사결정 지원 시스템입니다.

기존 악성 댓글 분류 모델을 기반으로 댓글의 독성 여부를 예측하고, 운영 정책과 유사 사례를 검색해 근거 기반 설명과 권장 액션을 제공합니다.

## Goals

- 기존 독성 댓글 분류 모델을 추론 가능한 파이프라인으로 분리
- 정책 문서와 유사 댓글 사례를 검색하는 RAG 구조 설계
- 댓글 검토 결과에 대해 정책 근거, 유사 사례, 권장 액션 제공

## Core Flow

```text
comment
-> toxicity classifier
-> policy retrieval
-> similar example retrieval
-> explanation generation
-> recommended action
```

## Tech Stack

- Python 3.13
- PyTorch
- scikit-learn
- pandas / NumPy
- gensim
- NLTK

FastAPI, vector search, and RAG dependencies will be added later when the service layer is implemented.

## Planned Output

```json
{
  "label": "toxic",
  "score": 0.91,
  "category": "insult",
  "reason_summary": "Direct personal attack was detected.",
  "matched_policies": [],
  "similar_examples": [],
  "recommended_action": "hide"
}
```

## MVP Scope

- Single comment moderation flow
- Toxicity score prediction
- Policy document indexing
- Similar example retrieval
- Rule-based action recommendation
