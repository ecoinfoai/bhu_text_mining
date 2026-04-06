# formative-analysis v0.13.0 — C3R Global Audit Summary

**Date**: 2026-04-06
**Scope**: 94 modules (42.7K LOC), 142 test files (82.7K LOC), 20 CLI commands
**Audit Team**: 8 specialist agents across 8 layers
**Test Artifacts**: 225 new tests written (168 adversary + 32 integration + 25 E2E)
**Status**: Discovery complete. Remediation pending user approval.

---

## Executive Summary

formative-analysis v0.13.0은 **보안 기반은 우수하나, 데이터 무결성에 체계적 결함**이 있는 프로젝트이다.

### C3R 종합 점수

| C3R 축 | 점수 | 핵심 근거 |
|--------|------|----------|
| **C**onsistency (일관성) | **7/10** | NaN 처리 불일치(3개 모듈), 비원자적 YAML 쓰기(7개 모듈), 로깅 패턴 혼재 |
| **C**omfortableness (사용성) | **7/10** | CLI help 양호하나, 8개 플래그 미문서화, 에러 메시지 복구 안내 부족 |
| **C**ontinuity (연속성) | **8/10** | 데드코드 최소, 순환 의존 0, 린트 0 위반, 중복 dataclass 1건 |
| **R**obustness (강건성) | **7/10** | 보안 우수, 168 adversary 전체 통과, 단 클래스/학기 격리 부재 |
| **종합** | **7.3/10** | |

### 결함 현황 총괄

| Severity | Count | 주요 내용 |
|----------|-------|----------|
| **Critical** | **2** | 중복 TopicTrendResult dataclass, eval-dir 포맷 불일치 |
| **High** | **14** | NaN 전파(5), 클래스/학기 격리 부재(2), 미문서화 플래그(2), raw dict 경계(1), 미사용 상수(1), deprecated 모듈 활성(1) |
| **Medium** | **25** | path traversal(3), 비원자적 쓰기(8), 점수 범위 미검증(4), 유니코드 미처리(3), 로깅 혼재(1), 기타(6) |
| **Low** | **22** | 포매팅(226파일), 데드코드(2), 타입힌트 갭(5), docstring 누락(10), 기타(5) |
| **합계** | **63** | |

---

## 상세 발견사항 (Severity 순)

### CRITICAL (2건) — 즉시 수정 필요

#### CRIT-1: 중복 TopicTrendResult dataclass (서로 다른 시그니처)
- **위치**: `longitudinal_report_data.py:77` vs `student_longitudinal_data.py:179`
- **문제**: 동일 이름의 dataclass가 두 모듈에서 **필드가 다르게** 정의됨
  - 전자: `kendall_tau: float` (non-optional)
  - 후자: `kendall_tau: float | None` + `interpretation: str` (추가 필드)
- **영향**: 잘못된 모듈에서 import 시 런타임 에러 또는 데이터 누락
- **C3R**: Consistency, Robustness
- **출처**: Phase 4 (Contract), Phase 7 (Integration)

#### CRIT-2: eval-dir 포맷 불일치 (파이프라인 출력 ≠ 리포트 입력)
- **위치**: `pipeline_evaluation.py` → `report_data_loader.py:load_all_student_data()`
- **문제**: 파이프라인은 per-student 파일(`res_lvl4_S001.yaml`)을 쓰지만, 리포트 로더는 통합 파일(`res_lvl4/ensemble_results.yaml`)을 기대
- **영향**: `forma-report`, `forma-report-professor`가 0명 학생을 반환하며 조용히 실패
- **C3R**: Robustness, Comfortableness
- **출처**: Phase 8 (E2E)

---

### HIGH (14건) — 우선 수정 대상

#### NaN 전파 관련 (5건)

| # | 위치 | 문제 | C3R |
|---|------|------|-----|
| H-01 | `longitudinal_report_data.py:~226` | `np.mean(week_scores)` — NaN 미필터링 → class_weekly_averages 오염 | Consistency |
| H-02 | `section_comparison.py:~187` | `np.mean(arr)` / `np.std(arr)` — NaN → SectionStats 오염 | Consistency |
| H-03 | `student_longitudinal_data.py:~232` | NaN ensemble_score → trend_slope OLS 오류 | Consistency |
| H-04 | `longitudinal_report_data.py:420` | `kendalltau()` NaN 반환을 float으로 저장 (None 변환 없음) | Robustness |
| H-05 | `student_longitudinal_data.py:232` | `spearmanr()` NaN 반환 동일 문제 | Robustness |

#### 데이터 격리 부재 (2건)

| # | 위치 | 문제 | C3R |
|---|------|------|-----|
| H-06 | `longitudinal_store.py` | 레코드 키에 class 필드 없음 → 다반 동일 학번 덮어쓰기 | Robustness |
| H-07 | `longitudinal_store.py` | 레코드 키에 semester 필드 없음 → 학기 간 데이터 오염 | Robustness |

#### 모듈 계약 관련 (5건)

| # | 위치 | 문제 | C3R |
|---|------|------|-----|
| H-08 | `pipeline_evaluation.py:389,438` | `_build_counseling_summary()` raw dict 반환, 스키마 미검증 | Robustness |
| H-09 | `report_generator.py` | docstring "deprecated"이나 pipeline에서 여전히 import | Continuity |
| H-10 | `ensemble_scorer.py:45` | `WEIGHTS_V2` 정의됨, 실제 v2 로직은 인라인 기본값 사용 → 괴리 | Consistency |

#### 문서화 관련 (2건)

| # | 위치 | 문제 | C3R |
|---|------|------|-----|
| H-11 | `cli-reference.md` | `--class`, `--week-config` (forma-eval 핵심 플래그) 미문서화 | Comfortableness |
| H-12 | `cli-reference.md` | `forma-report-student`, `-batch`, `-summary` 3개 커맨드 미문서화 | Comfortableness |

---

### MEDIUM (25건) — 계획적 수정 대상

| # | 영역 | 건수 | 요약 |
|---|------|------|------|
| M-01~03 | Path Traversal | 3 | student_id 미새니타이즈 (report_generator, cli_report_student, pipeline_evaluation) |
| M-04~11 | 비원자적 YAML 쓰기 | 8 | domain_coverage_analyzer(2), lecture_*(4), ocr_pipeline(1), student_longitudinal(1) |
| M-12~15 | 점수 범위 미검증 | 4 | LongitudinalStore에 NaN/Inf/음수/범위 초과 저장 허용 |
| M-16~18 | 유니코드 처리 | 3 | 전각 숫자 학번 분리, 제로폭 문자 개념 불일치, BOM 첫 필드 오류 |
| M-19 | 로깅 혼재 | 1 | pipeline_evaluation.py: print(50회) + logging 혼용 |
| M-20 | _esc() 누락 | 1 | ~25개 Paragraph에서 _esc() 미사용 (재확인 필요) |
| M-21 | 강의 비교 2단계 | 1 | forma lecture compare가 analyze 선행 필요하나 미문서화 |
| M-22 | argparse 플래그 순서 | 1 | forma-deliver --no-config가 subcommand 뒤에 오면 에러 |
| M-23 | 동시 쓰기 데이터 손실 | 1 | longitudinal_store last-writer-wins (flock은 있으나 merge 없음) |
| M-24 | 에러 메시지 비행동유도 | 6 | "Store contains no records" 등 복구 안내 없는 에러 |
| M-25 | config 예제 누락 | 1 | forma.json 예제 없음, week.yaml.example에 lecture 필드 누락 |

---

### LOW (22건) — 개선 권장

| 영역 | 건수 | 요약 |
|------|------|------|
| 포매팅 drift | 1 | 226/247 파일 (ruff format 일괄 적용으로 해결) |
| 데드코드 | 2 | `ocr_compare.run_comparison()`, `mecab_shim.install()` |
| 타입힌트 갭 | 5 | 일부 public 함수 리턴 타입 누락 |
| Docstring 누락 | 10 | 일부 public 함수 |
| mypy 미설치 | 1 | dev 의존성에 mypy 미포함 |
| SMTP 평문 비밀번호 | 1 | forma.json에 SMTP 비밀번호 평문 가능 |
| YAML merge key | 1 | `<<` merge key로 예상치 못한 필드 주입 가능 |
| 기타 | 1 | private `_` 함수가 `__all__`에 포함 (domain_coverage_analyzer) |

---

## 테스트 실행 결과 총괄

| Phase | 신규 테스트 | Pass | Fail | 발견 결함 |
|-------|-----------|------|------|----------|
| Phase 6: Adversary (168) | 168 | 168 | 0 | 설계 수준 4건 (silent-accept) |
| Phase 7: Integration (32) | 32 | 29 | 3 | NaN 전파 3건 확인 |
| Phase 8: E2E (25) | 25 | 21 | 4 | eval-dir 불일치, 강의 비교 2단계, argparse 순서 |
| **합계** | **225** | **218** | **7** | |

기존 테스트(pre-audit): 142 파일, ~2500+ 테스트 케이스 (변경 없음)

---

## C3R 축별 상세 평가

### Consistency (일관성) — 7/10

**강점**:
- 에러 메시지 100% 영어 (**Pass**)
- CLI 옵션 100% kebab-case (**Pass**)
- import 스타일 100% 절대 (**Pass**)
- 함수 네이밍 100% snake_case (**Pass**)
- 예외 클래스 100% 내장형 (**Pass**)
- yaml.safe_load 100% 사용 (**Pass**)

**약점**:
- NaN 처리: `risk_predictor`만 `nanmean` 사용, 나머지 3개 모듈은 `np.mean` (-2)
- YAML 쓰기: 7개 모듈은 원자적, 8개 모듈은 비원자적 (-0.5)
- 로깅: 40+ 모듈은 `logging`, 6개 모듈은 `print` 혼용 (-0.5)

### Comfortableness (사용성) — 7/10

**강점**:
- 출력 디렉터리 자동 생성 (**Pass**)
- 기본값 합리성 (**Pass**)
- 20개 CLI help 텍스트 일관된 영어 (**Pass**)
- 파일 덮어쓰기 보호 (**Pass**)

**약점**:
- 8개 CLI 플래그 문서 미동기화 (-1.5)
- 6개 에러 메시지에 복구 안내 없음 (-0.5)
- forma.json 예제 없음 (-0.5)
- 배치 명령 진행률 표시 없음 (-0.5)

### Continuity (연속성) — 8/10

**강점**:
- ruff lint 0 위반 (**Pass**)
- 순환 import 0건 (**Pass**)
- 데드코드 2건만 (94 모듈 중) (**Pass**)
- 테스트 코드 비율 1.94:1 (매우 우수)
- CI/CD: Python 3.11/3.12/3.13 매트릭스

**약점**:
- 중복 TopicTrendResult dataclass (-1)
- deprecated report_generator.py가 여전히 활성 (-0.5)
- 원자적 쓰기 구현 3벌 중복 (-0.5)

### Robustness (강건성) — 7/10

**강점**:
- 보안: YAML injection 방어, _esc() 적용, eval/exec 없음 (**Pass**)
- 파일시스템: 원자적 쓰기 + fcntl.flock + 백업 (**Pass**)
- 동시성: 파일 잠금으로 corruption 방지 (**Pass**)
- 입력 검증: InterventionLog 포괄적 검증 (**Pass**)
- LLM 장애: 3-call 중앙값 + 앙상블 (**Pass**)
- 대용량: 10K 레코드, 500명 학생 처리 (**Pass**)
- v0.11.3 CRITICAL 2건 모두 수정 확인 (**Pass**)
- 168 adversary 테스트 전체 통과 (**Pass**)

**약점**:
- 클래스/학기 격리 부재 → 데이터 오염 가능 (-1.5)
- NaN/Inf/범위 초과 점수 무검증 저장 (-1)
- Path traversal 3곳 (-0.5)

---

## 이전 감사 대비 개선 현황

### v0.11.3 감사 (2026-03-13) 발견사항

| 항목 | 상태 |
|------|------|
| CRIT-1: Path traversal in delivery_prepare | **FIXED** (3-layer defense) |
| CRIT-2: joblib deserialization arbitrary code | **MITIGATED** (type validation post-load) |

### v0.12.0 일관성 감사 (2026-03-13) 발견사항

| 항목 | 상태 |
|------|------|
| C-1: risk_predictor np.mean → np.nanmean | **FIXED** (_safe_nanmean 구현) |
| C-2: report_generator _esc() 미사용 | **FIXED** (esc() 적용) |
| C-3: 파일 I/O 패턴 비일관 | **PARTIALLY FIXED** (주요 store는 원자적, 8개 모듈 미적용) |
| C-5: NaN across modules | **PARTIALLY FIXED** (risk_predictor만, 3개 모듈 미적용) |

---

## 수정 우선순위 권고

### Tier 1: 즉시 수정 (데이터 무결성 위험)
1. **CRIT-1**: TopicTrendResult 통합 (하나의 정의로 통일)
2. **CRIT-2**: eval-dir 포맷 통일 또는 양방향 로더 구현
3. **H-01~05**: NaN 전파 수정 (`np.nanmean`, scipy NaN→None 변환)
4. **H-06~07**: LongitudinalStore 레코드 키에 class+semester 추가

### Tier 2: 다음 릴리스 (사용성 + 보안)
5. **M-01~03**: Path traversal 새니타이즈
6. **H-11~12**: CLI 문서 동기화
7. **M-12~15**: LongitudinalStore 점수 범위 검증
8. **M-24**: 에러 메시지에 복구 안내 추가

### Tier 3: 계획적 개선 (일관성 + 연속성)
9. **M-04~11**: 비원자적 YAML 쓰기 → io_utils.atomic_write_yaml 통일
10. **H-08~10**: 모듈 계약 정리 (raw dict→typed, WEIGHTS_V2, deprecated 모듈)
11. **L-01**: `ruff format` 일괄 적용
12. **L-05**: mypy dev 의존성 추가 + 단계적 타입 체크

---

## 산출물 목록

```
_workspace/
├── audit_plan.md                   ← 감사 계획
├── audit_layer1_static.md          ← Phase 1: 정적 분석
├── audit_layer2_contracts.md       ← Phase 4: 모듈 계약
├── audit_layer3_integration.md     ← Phase 7: 통합 검증
├── audit_layer4_e2e.md             ← Phase 8: E2E 테스트
├── audit_layer5_consistency.md     ← Phase 2: 일관성
├── audit_layer6_adversary.md       ← Phase 6: 적대적 테스트
├── audit_layer7_security.md        ← Phase 5: 보안
├── audit_layer8_comfortableness.md ← Phase 3: 사용성
└── audit_summary.md                ← 본 문서 (종합 보고서)

tests/ (신규 추가)
├── test_adversarial_audit_tier1.py   (20 tests)
├── test_adversarial_audit_tier2.py   (26 tests)
├── test_adversarial_audit_tier3.py   (25 tests)
├── test_adversarial_audit_tier4.py   (35 tests)
├── test_adversarial_audit_env.py     (37 tests)
├── test_adversarial_audit_cross.py   (25 tests)
├── test_integration_audit.py         (32 tests)
└── test_e2e_audit.py                 (25 tests)
```
