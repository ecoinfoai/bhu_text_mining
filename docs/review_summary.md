# v2.0 개선 계획서 종합 검증 리포트

작성일: 2026-03-06
검증 에이전트: Tester, Auditor, Adversary (7 personas)

---

## 1. 검증 결과 총괄

| 에이전트 | CRITICAL | HIGH | MEDIUM | LOW | 합계 |
|----------|----------|------|--------|-----|------|
| Auditor | 6 | 6 | 9 | 4 | 25 |
| Tester | 8 | 14 | 18 | 12 | 52 |
| Adversary | 5 | 14 | 10 | 3 | 32 |
| **합계** | **19** | **34** | **37** | **19** | **109** |

중복 제거 후 고유 이슈 약 45건. 아래는 **구현 전 반드시 해결해야 할 항목**을 카테고리별로 정리한 것입니다.

---

## 2. CRITICAL 이슈 — 구현 차단 항목 (Must Fix Before Phase 0)

### 2.1 설계 결함

| ID | 이슈 | 발견자 | 대응 방안 |
|----|------|--------|-----------|
| ARCH-001 | `TripletExtractionResult` dataclass 정의 누락 | Auditor | evaluation_types.py에 추가. `StudentGraph`와 통합 또는 별도 정의 |
| ARCH-003 | "결정론적 채점" 주장이 사실과 불일치 | Auditor | 설계 원칙 문구를 "채점 로직은 결정론적, 입력 추출은 LLM + 3-call 합의로 안정화"로 수정 |
| TYPE-001 | `GraphComparisonResult` vs `GraphMetricResult` 역할 혼동 | Auditor | GraphMetricResult는 비유향 그래프용으로 유지, GraphComparisonResult는 유향 트리플릿용. 문서에 명시적 구분 기술 |

### 2.2 하위 호환성 파괴

| ID | 이슈 | 발견자 | 대응 방안 |
|----|------|--------|-----------|
| BC-001 | `--skip-llm` 의미 변경으로 기존 스크립트 파괴 | Auditor | `--skip-llm`을 deprecated alias로 유지 + `--skip-feedback`, `--skip-graph` 신규 추가 |
| BC-002 | `DEFAULT_WEIGHTS` 키 변경으로 앙상블 테스트 전부 실패 | Auditor+Tester | knowledge_graph 유무에 따라 가중치 프리셋 분기: (1) v1 호환 모드 (2) v2 그래프 모드 |
| CFG-001 | knowledge_graph 없는 config에서 graph_f1 가중치 처리 미정의 | Auditor | graph_f1=None이면 v1 가중치로 자동 폴백 |

### 2.3 운영 안전성

| ID | 이슈 | 발견자 | 대응 방안 |
|----|------|--------|-----------|
| ADV-3-1 | API 재시도/backoff 로직 전무 | Adversary+Tester | llm_provider.py에 exponential backoff (max 3회) + timeout 추가 |
| ADV-3-4 | 종단 YAML atomic write 부재 | Adversary | tempfile + os.rename 패턴 적용. .bak 자동 백업 |
| ADV-6-1 | 학생 답변 외부 API 전송 — 개인정보 보호 고려 없음 | Adversary | 개인정보 영향 평가 수행, 학생 동의서 절차 문서화, 프롬프트에서 student_id 제외 확인 |

### 2.4 프롬프트 설계

| ID | 이슈 | 발견자 | 대응 방안 |
|----|------|--------|-----------|
| PROMPT-001 | 3개 프롬프트 템플릿 본문이 계획서에 없음 | Auditor | Planner B가 작성한 초안을 계획서에 통합. 출력 형식(JSON), relation 표현(동사형) 등 명시 |
| SEC-001 | 학생 답변 프롬프트 인젝션 위험 | Auditor | 학생 답변을 XML 태그로 감싸 구조적 분리 + 결과 validation 로직 추가 |

---

## 3. HIGH 이슈 — Phase 구현 시 반드시 포함

### 3.1 입력 검증 (Adversary 지적: 검증이 전혀 없음)

| ID | 이슈 | 대응 방안 |
|----|------|-----------|
| ADV-2-1 | 마스터 그래프 50 edges vs 200자 답변 -> 전학급 level_0 | config 로딩 시 edge 수 / answer_limit 비율 경고. calibration run 도구 제공 |
| ADV-2-2 | rubric_tiers 비합리적 임계값 (min_f1: 0.99) | min_graph_f1 상한 0.95 검증. 모범 답안 calibration 권장 |
| ADV-2-3 | 3시간 강의 녹취록 -> API 호출 폭발 | 녹취록 길이 상한 (10,000자) 또는 세그먼트 수 상한 설정 |
| ADV-1-4 | 빈 답변 silent skip -> 교수 인지 불가 | score=0, level="No Response" 명시적 생성. 미응답 학생 목록 출력 |

### 3.2 한국어 NLP 특수성

| ID | 이슈 | 대응 방안 |
|----|------|-----------|
| ADV-7-1 | 용언 활용형 ("유지" vs "유지한다") 정규화 불충분 | `_normalize_relation()`에 형태소 분석 기본형 변환 적용 |
| ADV-7-2 | 한자어/외래어 이표기 미지원 ("항상성" vs "homeostasis") | knowledge_graph 노드에 `aliases` 필드 추가 |

### 3.3 테스트 커버리지

| ID | 이슈 | 대응 방안 |
|----|------|-----------|
| TDD-001 | LLM JSON 파싱 실패 경로 테스트 5가지 시나리오 부재 | 최소 5개 비정상 응답 테스트 함수 작성 |
| TDD-002 | 3-call 합의 알고리즘 경계 테스트 부재 | 5가지 합의 시나리오 테스트 (전원 불일치, 과반수, 전원 일치 등) |
| TDD-016 | 종단 저장소 concurrent access 테스트 부재 | threading 기반 동시 쓰기 테스트 추가 |
| EC-016 | API rate limit 429 에러 처리 테스트 부재 | retry/backoff 구현 후 mock 기반 테스트 |

### 3.4 교육학적 타당성

| ID | 이슈 | 대응 방안 |
|----|------|-----------|
| ADV-4-1 | F1이 오개념 심각도를 미반영 | extra_edges 중 마스터와 모순되는 관계에 별도 페널티. wrong_direction_edges 수를 앙상블에 감점 요소로 반영 |
| ADV-1-1 | 모범 답안 복사 탐지 없음 | model_answer와의 n-gram 겹침 비율 > 0.95 시 "복사 의심" 플래그 |
| ADV-4-2 | 퍼지 매칭 threshold 0.80의 실증 근거 없음 | pilot 데이터로 threshold 민감도 분석 필수. Phase 1 이후 calibration 단계 추가 |

---

## 4. 개선 계획서 수정 사항 요약

### 계획서 구조 변경

```
Phase -1 (신규): 사전 준비
  - 개인정보 영향 평가 + 학생 동의 절차
  - llm_provider.py에 retry/backoff/timeout 추가
  - evaluation_io.py에 atomic write 패턴 적용
  - TripletExtractionResult dataclass 정의

Phase 0: 스키마 + 타입 (기존 + 보강)
  - knowledge_graph.nodes에 aliases 필드 추가
  - config 검증 함수 (edge 수 / answer_limit 비율, threshold 범위)
  - --skip-llm deprecated alias 정책

Phase 1: 트리플릿 추출 + 그래프 비교 (기존 + 보강)
  - _normalize_relation() 한국어 형태소 기본형 변환
  - 파싱 실패 시 해당 call을 빈 결과로 처리 (graceful degradation)
  - calibration run 도구 (모범 답안으로 달성 가능 F1 측정)
  - 프롬프트 인젝션 방어 (XML 태그 분리)

Phase 2: 시각화 (변경 없음)

Phase 3: 피드백 생성 (기존 + 보강)
  - 빈 응답 전용 피드백 템플릿
  - tier별 피드백 길이 조절 (level_3: 500자, level_0: 2000자)

Phase 4: 강의 녹취록 (기존 + 보강)
  - 녹취록 길이 상한 검증
  - segment 수 상한

Phase 5: 종단 저장소 (기존 + 보강)
  - atomic write + .bak 백업
  - file locking (fcntl.flock)
  - manual_override 필드

Phase 6: 파이프라인 통합 (기존 + 보강)
  - 가중치 프리셋 분기 (v1 호환 / v2 그래프 모드)
  - 빈 응답 명시적 처리 (silent skip 제거)
  - config 해시 기록 (배치 간 변경 탐지)
  - 학생 단위 체크포인트 (재시작 시 완료 학생 skip)

Phase 7: E2E 테스트 (기존 + 보강)
  - 14개 추가 테스트 함수 포함
```

### 앙상블 가중치 분기 전략

```yaml
# knowledge_graph 있는 config (v2 모드)
weights_v2:
  concept_coverage: 0.25
  graph_f1: 0.30
  rasch_ability: 0.15
  rubric_level: 0.15
  bertscore: 0.10
  misconception_penalty: -0.05  # wrong_direction 비례 감점

# knowledge_graph 없는 config (v1 호환 모드)
weights_v1:
  concept_coverage: 0.35
  llm_rubric: 0.30      # 기존 LLM 채점 유지
  rasch_ability: 0.15
  kg_node_recall: 0.10
  bertscore: 0.10
```

---

## 5. Adversary Red Team 종합 판정

> **Risk Score: 6/10**
>
> 핵심 알고리즘(BERT 추출 -> 코드 채점 -> LLM 해석)은 교육 측정학적으로 건전하다.
> 그러나 5가지 영역에 심각한 결함이 있다:
>
> 1. **장애 내성** — API 재시도, 체크포인트, atomic write 전무
> 2. **입력 검증** — 마스터 그래프 크기, threshold, 녹취록 길이 무검증
> 3. **개인정보** — 학생 답변 외부 전송에 대한 법적 대응 없음
> 4. **한국어 NLP** — 용언 활용, 이표기, 외래어 대응 부족
> 5. **실용성** — 교수 마스터 그래프 수동 생성 부담, CLI 진입 장벽
>
> 프로토타입으로는 가능하나, 200명 규모 실제 운영에는 위 결함 수정 필수.

---

## 6. 다음 단계 권장 순서

1. 개인정보 영향 평가 + 학생 동의 절차 확립
2. llm_provider.py 보강 (retry, timeout)
3. evaluation_io.py atomic write
4. 계획서 수정 반영 (위 내용 통합)
5. Phase 0 시작
