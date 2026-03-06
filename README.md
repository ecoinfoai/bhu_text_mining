# formative-analysis (FormA)

형성평가를 통해 학생의 이해도를 분석하고, 도메인 지식 그래프 매칭으로 학습 피드백 근거를 생산하는 CLI 도구입니다.

## 주요 기능

- **시험지 생성** (`forma-exam`) — QR 코드 포함 시험 PDF 생성
- **OCR 파이프라인** (`forma-ocr`) — 스캔 답안지 OCR 및 학생별 응답 결합
- **평가 파이프라인** (`forma-eval`) — 4-Layer 형성평가 분석
  - Layer 1: 개념 커버리지 + 트리플릿 그래프 비교
  - Layer 2: LLM 기반 코칭 피드백 생성
  - Layer 3: Rasch IRT 통계 분석
  - Layer 4: 앙상블 점수 + PDF 리포트
- **일괄 평가** (`forma-eval-batch`) — 다반 동시 평가

## 설치

```bash
# 개발 환경
uv sync --extra dev

# 전역 CLI 설치
uv tool install .
```

## 사용법

```bash
# 시험지 생성
forma-exam --config exam_config.yaml --output exams/

# OCR 스캔 → 텍스트 추출 → 학생별 결합
forma-ocr scan --image-dir scans/ --config config.yaml --output ocr_results/
forma-ocr join --ocr-dir ocr_results/ --output joined.yaml

# 단일 반 평가
forma-eval --config exam.yaml --responses responses.yaml --output results/ --provider gemini

# 다반 일괄 평가
forma-eval-batch --config exam.yaml --join-dir results/ \
  --join-pattern "anp_1{class}_final.yaml" --output eval_out/ --classes A B C D
```

## 설정

API 키 설정 (택 1):

```bash
# 환경 변수
export GOOGLE_API_KEY="your-key"

# 또는 설정 파일: ~/.config/formative-analysis/forma.json
{
  "naver_ocr": {"secret_key": "...", "api_url": "..."},
  "llm": {"provider": "gemini", "api_key": "your-key"}
}
```

## 개발

```bash
# 테스트 실행
uv run pytest tests/ -q

# 린트
uv run pylint src/forma/
```

## 프로젝트 구조

```
├── src/forma/              # 메인 패키지
│   ├── cli.py              # forma-exam CLI
│   ├── cli_ocr.py          # forma-ocr CLI
│   ├── pipeline_evaluation.py      # 평가 파이프라인
│   ├── pipeline_batch_evaluation.py
│   ├── triplet_extractor.py        # LLM 트리플릿 추출
│   ├── graph_comparator.py         # 그래프 비교 (퍼지 매칭)
│   ├── graph_visualizer.py         # 그래프 시각화
│   ├── feedback_generator.py       # 코칭 피드백 생성
│   ├── ensemble_scorer.py          # 앙상블 점수 산출
│   ├── concept_checker.py          # 개념 커버리지 분석
│   ├── config.py                   # 설정 관리
│   └── ...
├── tests/                  # 테스트
├── docs/                   # 문서
└── pyproject.toml
```
