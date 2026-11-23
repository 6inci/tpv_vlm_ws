# semantic_task_navigation

고정된 외부 RGB 카메라 영상과 자연어 작업지시를 이용하여 물류 작업환경의 객체와 공간 관계를 이해하고, 작업 목적에 적합한 접근 위치와 이동 경로를 생성하는 시스템.

> **현재 상태 (2026-11-17): v0.2.**
> - v0.2 변경: 의미적 하역장 → 배치 가능 영역 → 로봇 접근 영역 → 목표점 → 경로를 분리하고, 단계마다 실패 코드를 따로 기록한다.
> - 모든 좌표는 **IMAGE_SPACE_DEMO**(이미지 픽셀)다.
> - 그리퍼 도달 범위와 배치 표면은 **측정되지 않았다**. 그래서 접근·경로 결과는 가정 조건부이며, 로봇 작업 성공은 주장하지 않는다.
> - 관련 문서: [CHANGELOG](docs/CHANGELOG.md), [v0.1 실패 분석](docs/ANALYSIS_v0.1_destination_failure.md), [v0.2 검증](docs/VERIFICATION_v0.2_2026-11-17.md), [v0.1 검증](docs/VERIFICATION_2026-11-15.md)
> - v0.1 동결본: `archive/v0.1_2026-11-15/`

## 빠른 시작

```bash
cd /home/binci/vlm_ws/semantic_task_navigation
PY=/home/binci/miniconda3/envs/stnav_gui/bin/python

$PY run.py                                   # GUI + 참고 이미지
$PY run.py --mode live --camera 0            # GUI + 실시간 카메라
$PY run.py --mode image --headless --analysis-mode IMAGE_DEMO   # 모델 없이 전체 흐름 확인
$PY run.py --mode image --headless --analysis-mode LOCAL_ONLY   # 실제 모델로 분석
$PY run.py --mode replay --run-dir runs/<run_id>                # 저장된 실행 다시 보기
$PY -m pytest -q                             # 오프라인 테스트
```

- 자세한 사용법: [docs/RUNBOOK.md](docs/RUNBOOK.md)
- 구조: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- 좌표계: [docs/COORDINATE_FRAMES.md](docs/COORDINATE_FRAMES.md)
- 평가 계획: [docs/EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md)

## 실행 환경

모델과 GUI는 서로 다른 프로세스에서 실행된다.

| 역할 | 환경 | 비고 |
|---|---|---|
| GUI·오케스트레이션·지도·경로·저장 | `stnav_gui` (신규, Python 3.10) | PySide6 6.12, OpenCV headless 5.0 ([requirements.txt](requirements.txt)) |
| Qwen3-VL-4B / Grounding DINO 추론 | 기존 `qwen_vl` (변경 없음) | 별도 worker 프로세스, 로컬 캐시만 사용 |

- 12 GB GPU에서 두 모델을 동시에 올리면 OOM이 발생했다(실측).
- 그래서 GPU에는 한 번에 한 모델만 올리고, 필요할 때 교체한다.

## 처리 흐름 (지원 작업)

- `transport_and_place`: 예) "컨베이어 벨트에서 생산된 빨간색 큐브를 트럭이 위치한 하역장에 놓아줘."
- `navigate_to_region`: 예) "트럭이 있는 하역장으로 이동해."

```text
명령 → 작업 해석(LLM, 스키마 검증) → 장면 이해(VLM)
→ 후보 탐지(DINO) → 크롭 검증(VLM)
→ 빨간 큐브 정밀 확인(ROI 확대 + HSV + 검증)
→ 공간 관계 → 지도(사전 지도 + 이번 프레임 장애물)
→ 의미적 하역장(기하 띠 + 구조 grounding 후보, 후보별 VLM 판정; 지도·정답은 모델에 주지 않음)
→ 배치 가능 영역(의미 영역 − 물체·장애물, 대상 크기 침식)
→ 로봇 접근 영역(reach 이내 ∩ 원점에서 도달 가능) → 목표점(경로 비용 + 거리)
→ A* + 직선화 → 픽셀 수준 경로 검증
→ 인계 요청(시뮬레이션) → runs/<run_id>/ 저장
```

## 폴더

| 위치 | 내용 |
|---|---|
| `run.py` | 진입점 (GUI / headless / replay) |
| `config/` | `app.yaml`, `models.yaml`(모델 ID·프롬프트·임계값), `camera.yaml`, `planning.yaml`, `maps/`, `demo/` |
| `src/semantic_task_nav/agent` | 명령 해석, 오케스트레이터(상태 머신) |
| `src/semantic_task_nav/perception` | 카메라 입력, 모델 worker/client, 탐지·검증, 큐브 검사, scene graph |
| `src/semantic_task_nav/semantics` | 하역 영역 추론 |
| `src/semantic_task_nav/geometry` | 좌표 변환, 보정 호환성 판정 |
| `src/semantic_task_nav/navigation` | 지도, 접근 위치, A*, 경로 검증 |
| `src/semantic_task_nav/robot` | 인계 요청과 시뮬레이션 상태 머신 |
| `src/semantic_task_nav/gui` | PySide6 GUI, 백그라운드 worker |
| `src/semantic_task_nav/visualization.py`, `artifacts.py` | 레이어·논문 그림, 실행 폴더 저장 |
| `scripts/evaluate_run.py` | 사람이 만든 정답과 비교해 항목별로 평가 |
| `tests/` | 단위·통합·GUI·목적지 체인 테스트 (`STNAV_RUN_MODEL_TESTS=1`: 실제 모델 테스트) |
| `archive/v0.1_2026-11-15/` | v0.1 동결 스냅샷 (수정 금지, 독립 실행) |
| `data/reference/` | GUI에서 저장한 사람 정답 (평가 전용, 아직 없음) |
| `runs/` | 실행별 결과 (새 폴더만 만들고 덮어쓰지 않음) |

## 주요 제한 (상세: VERIFICATION 문서)

- **좌표**
  - 이미지 픽셀 좌표만 사용한다.
  - 로봇 크기와 도달 거리는 픽셀 단위로 정한 **가정값**이다.
- **지도**
  - 지도는 기존 zone 초안을 옮긴 것이다(`DRAFT_UNCONFIRMED`).
  - 사람이 검토해야 한다.
- **참고 이미지(개발 데이터)에서 실제 모델을 실행한 결과**
  - v0.1: VLM이 트럭 위쪽 바닥을 골라 `SEMANTIC_REGION_INACCESSIBLE`
  - v0.2: 모델이 찾은 주차 구획이 하역장으로 판정되어 계획까지 완료됐다.
    - 다만 배치 영역에서 가장 가까운 로봇 위치까지 155 px라서, 도달 거리 가정(180 px) 아래에서만 성립한다.
- **사람 정답**: 하역장과 배치 영역의 사람 정답이 아직 없다. GUI `정답:` 유형으로 입력할 수 있다.
- **카메라**
  - 현재 연결된 카메라는 노트북 내장 카메라다.
  - 보정 대상인 외부 HD Web Camera가 아니므로 라이브 모드에서는 왜곡 보정 보기도 비활성이다.

## 기존 자산 재사용 원칙

- 모델은 기존 Hugging Face 캐시를 `local_files_only`로 읽는다. 복사하거나 다시 다운로드하지 않는다.
- A* 핵심 루프와 Qwen·DINO 로딩 방식은 기존 코드에서 옮겨 왔다. 출처는 각 파일 docstring에 적었다.
- 기존 연구 폴더, 보정 파일, conda 환경은 읽기만 했다.
