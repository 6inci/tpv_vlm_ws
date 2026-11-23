# RUNBOOK — 실행 설명서

모든 명령은 프로젝트 폴더에서 실행한다.

```bash
cd /home/binci/vlm_ws/semantic_task_navigation
PY=/home/binci/miniconda3/envs/stnav_gui/bin/python
```

`conda activate stnav_gui` 후 `python`을 써도 된다.

## 1. 최초 실행 확인

```bash
$PY -m pytest -q                                   # 오프라인 테스트 (약 45초)
$PY run.py --mode image --headless --analysis-mode IMAGE_DEMO
```

두 번째 명령은 모델 없이 전체 흐름을 확인한다.
- 결과로 `PLAN_COMPLETE`와 `runs/<run_id>/` 경로가 출력되어야 한다.
- IMAGE_DEMO는 사람이 입력한 데이터를 쓴다(`config/demo/`). 결과에는 `[MANUAL]`로 표시된다.

실제 모델 점검:

```bash
STNAV_RUN_MODEL_TESTS=1 $PY -m pytest -q tests/test_models_integration.py   # 약 20초
```

## 2. GUI 실행

```bash
$PY run.py                                   # 기본 참고 이미지
$PY run.py --mode image --image <파일>
$PY run.py --mode video --video <파일>
$PY run.py --mode live [--camera 0]
$PY run.py --mode replay --run-dir runs/<run_id>
```

`--analysis-mode`, `--command "<명령>"`, `--map <지도.json>` 옵션도 함께 쓸 수 있다.

### 화면 구성

**왼쪽 위 — CAMERA VIEW**
- 선택한 입력이 실시간으로 표시된다.
- 분석 중에도 계속 갱신된다(backpressure로 최신 프레임만 표시).

**왼쪽 아래 — RESULT VIEW**
- `화면` 목록에서 11개 화면을 고른다.
  - RAW, UNDISTORTED, OBJECT_DETECTION, RED_CUBE_ROI, OBJECT_RELATIONSHIP, SEMANTIC_LOADING_REGION, TRAVERSABLE_MAP, OCCUPANCY_MAP, PICKUP_PATH, DELIVERY_PATH, FINAL_RESULT
- 체크박스로 레이어를 켜고 끈다.
- 결과는 **분석 시점에 캡처한 프레임** 위에 그린다.
- 제목 줄에 프레임 번호와 시각, 그리고 "현재 라이브 아님"이 표시된다.

**오른쪽 패널**

| 영역 | 내용 |
|---|---|
| USER COMMAND | 모드, 입력 소스, 명령, `분석 시작` / `중지` / `모델 준비` |
| AGENT ANALYSIS | 단계별 상태, 모델, 입력·출력 요약, 시간, 오류, 다음 단계 (실제 이벤트 기록) |
| DETECTION RESULTS | 후보·채택 객체, score, bbox(raw px), 출처, VLM 검증 결과 |
| 작업 인계 | 시뮬레이션 확인 버튼 (실제 로봇 명령은 비활성) |

**하단**
- 상태 표시줄: Qwen3-VL, Grounding DINO, Agent, Camera, Map, Planner 상태와 FPS, 분석 시간, GPU(nvidia-smi 실측), 오류
- 버튼: `실행 결과 저장`, `논문용 이미지 저장`, `이전 실행 불러오기`

### 명령 입력과 분석

1. 명령창에 자연어로 입력한다. 기본 예시가 들어 있다.
2. 모드를 고른다.
   - `LOCAL_ONLY`: 기본값. 로컬 Qwen3-VL과 DINO를 사용한다.
   - `LOCAL_VLM_WITH_API_AGENT`: 명령 해석만 API를 쓴다(§5).
   - `IMAGE_DEMO`: 모델 없이 수동 데이터를 쓴다.
3. `분석 시작`을 누른다. 그 시점의 프레임이 고정되고, 프레임 번호와 시각이 저장된다.
4. 시간
   - 처음에는 모델 로드에 수 초가 걸린다.
   - 참고 이미지의 전체 분석은 약 80초 걸렸다(실측).
   - `모델 준비`로 Qwen을 미리 올려 둘 수 있다.
5. 분석 중 다시 `분석 시작`을 누르면 거부된다. 중복 실행은 막혀 있다.
6. `중지`를 누르면 실행 중인 추론 프로세스가 종료된다.
   - 부분 결과는 `CANCELLED`로 저장된다.
   - 다음 분석에서 모델을 다시 로드한다.

라이브·동영상 모드에서는 분석 프레임과 현재 프레임의 이동량을 1초마다 계산한다. 기준을 넘으면 "⚠ 장면 변화 — 결과 재사용 주의"가 표시된다.

### 지도 편집 (사람이 검토하는 지도)

1. `지도 편집`을 켠다.
2. 유형을 고른다: 통행 가능 영역, 고정 장애물, 금지 영역, 시작점.
3. 결과 화면을 클릭해 꼭짓점을 찍는다.
4. `다각형 완료`를 누른다.
5. `지도 저장`을 누르면 새 파일로 저장된다(덮어쓰기 금지).
   - 편집에 쓴 프레임이 `<이름>.reference.png`로 함께 저장된다.
   - 이 프레임은 이후 카메라 정렬 검사의 기준이 된다.
6. 편집한 지도는 `EDITED_IN_GUI_UNCONFIRMED`(수동 주석)로 기록되고, 모델 출력과 구분해서 표시된다.

### 사람 정답 영역 (평가용)

1. `지도 편집`을 켜고 유형을 고른다.
   - `정답: 하역장(사람)`: 다각형 1개
   - `정답: 배치영역(사람)`: 다각형 여러 개 가능
2. 다각형을 그리고 `다각형 완료`를 누른다.
3. `정답 저장`을 누른다.
   - `data/reference/*.json`으로 저장된다(덮어쓰기 금지, 작성자 입력).
   - 이미지 SHA-256이 함께 기록된다.
4. 정답은 `reference_regions` 레이어에만 그려진다. 흰 점선과 `[HUMAN]`으로 표시된다.
5. 정답은 **분석 입력, 모델, 지도, 실행 기록에 들어가지 않는다.**
6. 채점: `scripts/evaluate_run.py --gt data/reference/<파일>.json runs/<id>`
   - 정답에 없는 항목은 `NO_GT (pending)`으로 남는다.
7. 지도의 `semantic_reference_zones`(예: Z06)는 미확정 개발 초안이다. `prior_zones` 레이어로 따로 표시한다.

### 작업 인계 (시뮬레이션)

- 분석이 `PLAN_COMPLETE`이면 상태가 `WAITING_FOR_PICKUP_CONFIRMATION`이 된다.
- `픽업 성공 확인`을 누르면 다음이 바뀐다.
  - 상태: `WAITING_FOR_PLACEMENT_CONFIRMATION`
  - 경로 B: `PLANNED_ASSUMING_PICKUP` → `ACTIVE_AFTER_CONFIRMED_PICKUP`
- 모든 확인은 `handoff/state_NNN.json`과 `handoff/events.jsonl`에 새로 추가된다.
- 확인 없이 완료로 바뀌는 경로는 없다.

## 3. 카메라 선택

- GUI에서 입력을 `LIVE CAMERA`로 바꾸고 장치를 고른 뒤 `열기`를 누른다.
- 장치 목록은 `/sys/class/video4linux`에서 읽는다.
- CLI에서는 `--camera 0` 또는 `--camera /dev/video0`을 쓴다.
- 기본값은 `config/camera.yaml`의 `live.device`다. null이면 첫 장치를 쓴다.
- 해상도와 FOURCC는 `config/camera.yaml`에서 정한다(1920×1080, MJPG).
- 현재 연결된 장치는 `/dev/video0 HP True Vision FHD Camera`(노트북 내장)다.
  - 보정 파일의 카메라(`HD Web Camera`)와 이름이 달라서 왜곡 보정 보기가 비활성이다.
  - 지도의 기준 이미지와 장면이 다르면 `MAP_CAMERA_MISALIGNED`로 경로계획이 거부된다.
  - 외부 고정 카메라를 연결한 뒤 지도를 검토하거나 새로 편집해야 한다.

## 4. 로컬 모델 설정 (`config/models.yaml`)

- `worker.python`: 추론 환경의 Python이다. 기존 `qwen_vl`을 수정 없이 사용한다.
- `qwen.model_id`, `grounding_dino.model_id`: HF 캐시의 모델 ID다. `local_files_only=True`로 읽고, worker 환경에는 `HF_HUB_OFFLINE=1`이 설정된다.
- `worker.gpu_residency: exclusive`: GPU에는 한 번에 한 모델만 올린다.
  - 12 GB GPU에서 Qwen3-VL-4B(bf16, 8.3 GiB)와 DINO 추론을 동시에 하면 CUDA OOM이 났다(실측).
- 임계값, 클래스 프롬프트, 검증 설명, 프롬프트 문구는 모두 같은 파일에 있다.
  - 프롬프트 id와 sha256은 manifest에 기록된다.

## 5. API 선택 (선택 사항)

- `LOCAL_VLM_WITH_API_AGENT` 모드는 **명령 해석만** OpenAI 호환 API로 보낸다. 텍스트만 보내고 이미지는 보내지 않는다.
- 설정 방법
  - `cp .env.example .env`
  - `.env`에 `OPENAI_API_KEY=` 값을 넣는다. `.env`는 git 추적에서 제외된다.
  - `config/app.yaml`의 `api_agent`에서 모델과 주소를 정한다.
- 키가 없거나 요청이 실패하면 이벤트로 기록한 뒤 로컬 Qwen으로 해석한다(`task_parse.json`의 `api_fallback`).
- 키 값은 로그나 결과에 기록하지 않는다.
- **실제 API 호출은 시험하지 않았다(NOT_TESTED).**

## 6. 결과 저장 위치

실행할 때마다 `runs/<YYYYmmdd_HHMMSS>_<hex>/`가 새로 생긴다.

```text
input/      raw_frame.png (디코딩된 원본 픽셀), command.txt, frame.json
analysis/   task_parse.json, scene_analysis.json, detections.json, target_inspection.json,
            semantic_regions.json, agent_events.jsonl, model_calls.jsonl, crops/, roi/,
            region_select_input.png, undistorted_view.png (보정 호환 시)
planning/   map.json, pickup_path.json, delivery_path.json, validation.json
handoff/    *_request.json, state_001.json …, events.jsonl
figures/    01_raw … 10_traversable_map.png (원본 해상도 + 범례 띠)
result.json, manifest.json
```

- **result.json**: 전체 결과다. replay에도 이 파일을 쓴다.
- **manifest.json**: 재현 정보를 담는다.
  - 입력 SHA-256, 코드 해시, 모델 ID와 commit, 프롬프트 sha256, 추론 파라미터
  - 보정 판정, 지도와 정렬 검사, 시간, 수동/자동 구분, 설정 스냅샷, 파일 해시
- **추가 저장**
  - `실행 결과 저장`: `gui_exports/`에 GUI 스크린샷과 현재 화면을 저장한다.
  - `논문용 이미지 저장`: `figures_export_<시각>/`에 그림을 다시 내보낸다.
- **replay 검증**: `$PY run.py --mode replay --run-dir runs/<id> --headless`를 실행하면 저장된 그림과 다시 렌더링한 그림을 픽셀 단위로 비교한다.
- **평가**: `$PY scripts/evaluate_run.py --gt <정답.json> runs/<id> ...` (형식은 EXPERIMENT_PLAN 참고)

### 목적지 체인 확인 (v0.2)

- 화면 `DESTINATION_CHAIN` 또는 그림 `11_destination_chain.png`에서 다음을 각각 볼 수 있다.
  - 의미적 하역장(자홍), 배치 영역(노랑), 로봇 접근 영역(청록 셀), 목표점, 경로 B
- 기록 파일
  - `analysis/semantic_regions.json`: 후보, 구조 grounding 원문, 후보별 판정 이미지와 원문(`region_judge_cand*.png`)
  - `analysis/placement_regions.json`
  - `analysis/destination_chain.json`: 단계 표, `goal_reach_check`, claims
- ablation: `run.py ... --set 'planning.destination.candidate_sources=["geometric_bands"]'`. manifest의 `config_overrides`에 기록된다.

## 7. 오류 발생 시 확인할 내용

| 증상 | 확인 |
|---|---|
| `MODEL_WORKER_PYTHON_MISSING` / worker 시작 실패 | `config/models.yaml`의 `worker.python` 경로, `runs/_worker_logs/worker_*.log` |
| `MODEL_CUDA_OOM` | 다른 GPU 프로세스(`nvidia-smi`). `gpu_residency: exclusive` 유지 |
| `MODEL_TIMEOUT` | `worker.request_timeout_s`. 시간 초과 시 worker를 종료하고 다음 분석에서 다시 시작한다 |
| `INVALID_JSON` / `SCHEMA_INVALID` (명령 해석) | `analysis/task_parse.json`의 `raw_response`. 잘못된 계획을 만들지 않고 중단한다 |
| `TARGET_NOT_FOUND` / `TARGET_AMBIGUOUS` | `figures/03_red_cube_roi.png`, `analysis/target_inspection.json`의 후보별 VLM 응답 |
| `NO_TRUCK`, `NO_LOADING_REGION_FOUND`, `SEMANTIC_REGION_UNCERTAIN` | `analysis/semantic_regions.json`의 후보별 `vlm_plausible`·`vlm_reason`, `region_judge_cand*.png`, `structure_grounding` |
| `PLACEMENT_REGION_EMPTY` | `analysis/placement_regions.json`의 `areas_px`, `excluded` |
| `ROBOT_APPROACH_NOT_FOUND` / `ROBOT_APPROACH_UNREACHABLE` | `destination_chain.json`의 `chain`, `nearest_valid_cell_to_anchor_px`. reach는 가정값(`planning.yaml`) |
| `MAP_CAMERA_MISALIGNED` | 카메라가 움직였거나 다른 카메라다. 지도를 다시 편집한다 |
| `START_IN_COLLISION` / `NO_ACCESSIBLE_CANDIDATE` / `NO_PATH` | `figures/05_occupancy_map.png`. `planning.yaml`의 로봇 반경과 도달 거리(픽셀 가정값) |
| `DEMO_INPUT_IMAGE_MISMATCH` | IMAGE_DEMO 수동 데이터는 참고 이미지 전용이다(SHA-256 확인) |
| 카메라 `DISCONNECTED` | 장치 번호, 다른 프로그램의 점유 여부. IMAGE 모드는 계속 사용할 수 있다 |
| pytest가 `launch_testing` 등 ROS 플러그인 오류 | `pytest.ini`가 ROS 플러그인을 끈다. 프로젝트 폴더에서 실행한다 |
| GUI가 뜨지 않음 | `stnav_gui` 환경 확인. `qwen_vl` 환경에는 PySide6가 없다(의도된 분리) |
