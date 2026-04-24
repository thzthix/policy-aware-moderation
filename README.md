# Policy-Aware Moderation

커뮤니티 운영자를 위한 정책 기반 댓글 모더레이션 의사결정 지원 시스템입니다.

현재 Phase 1에서는 정책 검색이나 설명 생성보다, **독성 댓글 분류 baseline을 비교하고 단일 댓글 추론 경로를 안정화하는 것**에 집중합니다.

## Current Scope

- 단일 댓글 독성 여부 예측
- 모델 artifact 저장 및 로딩
- 공개 데이터셋 기반 비교 실험
- threshold 및 하이퍼파라미터 튜닝

아직 하지 않는 것:

- FastAPI
- RAG
- vector DB
- UI

## Final Baseline Choice

현재 비교 결과 기준 최종 운영 후보는 다음 모델입니다.

```text
char n-gram TF-IDF + Logistic Regression
```

선택 이유:

- 한국어 욕설 변형, 자모 반복, 오타, 붙여쓰기를 잘 포착함
- `Word2Vec + GRU`, `FastText + GRU`보다 높은 ROC AUC / F1을 기록함
- 구조가 단순하고 해석 가능성이 높음

현재 추천 설정:

```text
analyzer = char
ngram_range = (2, 4)
class_weight = "balanced"
C = 2.0 근처
threshold = 0.47 근처
```

## Experiment Summary

비교한 실험 축:

1. `Word2Vec + GRU`
2. `FastText + GRU`
3. `char n-gram TF-IDF + Logistic Regression`

핵심 결론:

> 이 문제는 semantic understanding보다 character-level pattern recognition 성격이 강해서, char n-gram 기반 TF-IDF가 가장 잘 맞았습니다.

## Inference Paths

### GRU baseline

```text
comment
-> clean_text
-> tokenize_text
-> sequence embedding
-> padding
-> GRU
-> sigmoid
-> label / score
```

### TF-IDF final candidate

```text
comment
-> clean_text
-> char n-gram TF-IDF
-> Logistic Regression
-> label / score
```

## Artifact Files

TF-IDF + Logistic Regression artifact:

```text
artifacts/tfidf_logreg/tfidf_vectorizer.pkl
artifacts/tfidf_logreg/logistic_regression.pkl
artifacts/tfidf_logreg/model_config.json
```

GRU baseline artifact:

```text
artifacts/classifier/word2vec.model
artifacts/classifier/best_model.pth
artifacts/classifier/model_config.json
```

## Commands

TF-IDF + Logistic Regression 평가:

```bash
python3 scripts/evaluate_tfidf_logreg.py \
  --train-csv data/kmhas_train.csv \
  --val-size 0.2 \
  --threshold 0.47 \
  --ngram-min 2 \
  --ngram-max 4 \
  --c 2.0 \
  --use-class-weight
```

TF-IDF + Logistic Regression artifact 생성:

```bash
python3 scripts/train_tfidf_logreg.py \
  --train-csv data/kmhas_train.csv \
  --output-dir artifacts/tfidf_logreg \
  --threshold 0.47 \
  --ngram-min 2 \
  --ngram-max 4 \
  --c 2.0 \
  --use-class-weight
```

TF-IDF + Logistic Regression 단일 댓글 예측:

```bash
python3 scripts/predict_tfidf_logreg.py \
  --artifact-dir artifacts/tfidf_logreg \
  --comment "너 진짜 최악이다"
```
