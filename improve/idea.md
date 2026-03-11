# FormA 개선 아이디어

## 배경

한 학기 동안 매주 형성평가를 진행하며 FormA를 운용하는 과정에서 식별된 개선 사항들.

---

## 1. 설정 파일 체계 정비 (핵심)

### 현황 문제
- `forma.yaml`에 학기 고정 값과 주차별 변동 값이 혼재함
- CLI마다 설정 로딩 방식이 제각각 (불일치)

| CLI | forma.yaml 자동 읽기 | 주차별 yaml |
|-----|---------------------|-------------|
| `forma-ocr` | ✅ `apply_project_config()` | `--config ocr_scan.yaml` |
| `forma-eval` | ❌ 읽지 않음 | `--eval-config eval_w1_A.yaml` |
| `forma-exam` | ❌ 읽지 않음 | `--config questions.yaml` |

### 제안: 3계층 통합 구조

```
CLI 플래그  >  week.yaml (주차별)  >  forma.yaml (학기별)  >  기본값
```

모든 CLI가 동일한 우선순위 규칙으로 설정을 읽음.
`week.yaml` 하나가 `forma-select`, `forma-ocr`, `forma-eval` 세 CLI의 주차 설정을 모두 담당.

---

### `forma.yaml` (학기 고정 — 프로젝트 루트에 1개)

```yaml
project:
  course_name: "인체구조와기능"
  year: 2026
  semester: 1
  grade: 1

classes:
  identifiers: [A, B, C, D]

ocr:
  naver_config: ""          # 비워두면 /run/agenix/forma-config 자동 사용
  credentials: ""           # Google Sheets OAuth2 인증 파일 경로
  spreadsheet_url: ""       # Google Sheets URL (join 우선 사용)

evaluation:
  provider: "gemini"
  n_calls: 3

paths:
  longitudinal_store: "longitudinal_store.yaml"
  font_path: null

reports:
  dpi: 150
```

### `week.yaml` (주차별 — 주차 폴더 안에 1개)

```yaml
# 형성평가_1주차_결과/week.yaml
week: 1

# forma-select / forma-exam 설정
select:
  source: "../../형성평가_문제초안/Ch01_서론_FormativeTest.yaml"
  questions: [1, 3]           # 출제할 문제 sn 번호
  num_papers: 220
  form_url: "https://docs.google.com/.../viewform?entry.XXXX={student_id}"
  exam_output: "인구기_형성평가_1주차.pdf"

# forma-ocr 설정 (분반별 경로는 {class} 패턴으로 자동 해석)
ocr:
  num_questions: 2
  image_dir_pattern: "인구기_형성평가_1{class}_1주차"
  ocr_output_pattern: "인구기_형성평가_1{class}_1주차/ocr_results.yaml"
  join_output_pattern: "인구기_형성평가_1{class}_1주차/final.yaml"
  join_forms_csv: "google_forms_1주차_응답.csv"   # Sheets 실패 시 폴백
  crop_coords:                 # 최초 클릭 후 자동 저장됨 (아래 §3 참고)
    - [x1, y1, x2, y2]        # 1번 문제 영역
    - [x1, y1, x2, y2]        # 2번 문제 영역

# forma-eval 설정
eval:
  config: "../../형성평가_문제초안/Ch01_서론_FormativeTest.yaml"
  questions_used: [1, 3]
  responses_pattern: "인구기_형성평가_1{class}_1주차/final.yaml"
  output_pattern: "인구기_형성평가_1{class}_1주차/eval/"
  skip_feedback: false
  skip_graph: false
  generate_reports: true
```

### 실행 방식

```bash
cd 형성평가_1주차_결과

forma-select                   # week.yaml 자동 탐색 → questions.yaml + PDF 생성
forma-ocr scan --class A       # week.yaml + {class}=A 자동 해석
forma-ocr join --class A       # week.yaml + {class}=A 자동 해석
forma-eval --class A           # week.yaml + {class}=A 자동 해석
```

---

## 2. `forma-select` 신규 명령 추가

### 현황
- 형성평가 문제 초안(`Ch01_서론_FormativeTest.yaml`)에 5개 문제가 있고 매주 일부 선택 출제
- 현재는 수동으로 `questions.yaml` 작성 필요
- 출처 파일과 선택된 `sn` 번호가 `questions.yaml`에 기록되지 않음

### 동작
1. `week.yaml`의 `select` 섹션을 읽음
2. `source` 파일에서 지정한 `sn` 문제를 추출
3. `questions.yaml` 생성 (출처·선택 번호 기록)
4. `--generate-pdf` 또는 `week.yaml`의 `exam_output` 지정 시 시험지 PDF 즉시 생성

### 생성되는 `questions.yaml` 형식

```yaml
source: "../../형성평가_문제초안/Ch01_서론_FormativeTest.yaml"
selected_sn: [1, 3]
week: 1
num_papers: 220
form_url: "https://..."
questions:
  - topic: 개념이해
    text: "생체항상성..."
    limit: 200자 내외
  - topic: 적용
    text: "출산 과정..."
    limit: 200자 내외
```

---

## 3. 크롭 좌표 자동 저장

### 현황 문제
- `crop_coords`가 없으면 matplotlib 창이 열리고 클릭으로 좌표 수집
- 수집된 좌표가 **yaml에 저장되지 않음** → 매 실행 시 다시 클릭해야 함

### 제안
- 인터랙티브 클릭 후 수집된 좌표를 `week.yaml`의 `ocr.crop_coords`에 자동으로 기록
- 이후 실행 시 저장된 좌표를 재사용 (클릭 불필요)
- 강제 재측정: `forma-ocr scan --class A --recrop`

---

## 4. `bhu-*` 잔재 정리

`pyproject.toml` 진입점은 이미 전부 `forma-*`로 등록되어 있음.
`cli.py`, `cli_ocr.py`의 docstring/주석에만 구버전 이름이 남아 있어 수정 필요:

- `cli.py`: `bhu-exam` → `forma-exam`
- `cli_ocr.py`: `bhu-ocr` → `forma-ocr`

---

## 5. 전체 워크플로우 (1주차 예시)

```bash
# 0. 프로젝트 초기화 (학기 시작 시 1회)
cd /home/kjeong/Documents/anp2026_formative_analysis
forma-init

# 1. 문제 선택 및 시험지 생성 (week.yaml 기반)
cd 형성평가_1주차_결과
forma-select
# → questions.yaml 생성 + 인구기_형성평가_1주차.pdf 생성

# 2. 답안지 jpg 압축 해제 후 OCR (분반별)
forma-ocr scan --class A
# crop_coords 없으면 클릭 입력 → week.yaml에 자동 저장
# 이후 B, C, D는 저장된 좌표 재사용

# 3. OCR 결과 + Google Forms 조인 (분반별)
forma-ocr join --class A
# spreadsheet_url(forma.yaml) 우선 → 실패 시 join_forms_csv(week.yaml) 폴백

# 4. 평가 실행 (분반별)
forma-eval --class A
```

---

## 6. `forma.yaml`에서 제거할 항목

주차별로 달라지므로 `week.yaml`로 이동:

| 항목 | 이동 대상 |
|------|-----------|
| `ocr.num_questions` | `week.yaml` → `ocr.num_questions` |
| `paths.exam_config` | `week.yaml` → `eval.config` |
| `paths.join_dir` | `week.yaml` → `ocr.image_dir_pattern` |
| `paths.output_dir` | `week.yaml` → `eval.output_pattern` |
| `classes.join_pattern` | `week.yaml` → `ocr.join_output_pattern` |
| `classes.eval_pattern` | `week.yaml` → `eval.output_pattern` |
| `current_week` | `week.yaml` → `week` |
| `evaluation.skip_*` | `week.yaml` → `eval.skip_*` |
| `reports.skip_llm`, `reports.aggregate` | `week.yaml` → `eval` 섹션 |
