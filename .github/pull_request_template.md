# PR 제목

<!-- feat: predictor 구현 / refactor: preprocessing 분리 -->

---

## 변경 목적 (Why)

*

---

## 변경 내용 (What)

*
*

---

## Scope Check

* [ ] PR이 하나의 목적을 가진다
* [ ] 불필요한 변경이 포함되지 않았다

---

## 검증 방법 (How to test)

### 실행한 테스트

* [ ] focused test 실행
* [ ] related module test 실행
* [ ] full test suite 실행
* [ ] manual check 실행
* [ ] 실행하지 못한 테스트가 있다면 이유를 작성했다

### 테스트 명령

```bash
# 예시
pytest tests/test_preprocessing.py
pytest tests/test_predictor.py
pytest
python scripts/predict_comment.py
```

### 테스트 케이스

* [ ] 정상 입력
* [ ] 빈 문자열
* [ ] invalid input
* [ ] OOV-only 입력
* [ ] missing files/artifacts

---

## 결과

```json
{
  "label": "toxic",
  "score": 0.91
}
```

---

## 영향 범위 (Impact)

* [ ] 기존 기능 영향 없음
* [ ] 일부 로직 변경 있음 (설명 필요)

## 설명:

---

## 설계 판단 / 고려 사항

*

---

## Known Limitations

*

---

## 기본 규칙 체크

* [ ] 하드코딩된 경로 없음
* [ ] debug print 없음
* [ ] 함수가 과도하게 길지 않음
* [ ] training / inference / evaluation 분리 유지
* [ ] 한 파일 한 책임 원칙 유지

---

## TODO

*
