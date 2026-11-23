# ARCHITECTURE (v0.1, 2026-11-15)

## 1. 프로세스 구성

```text
┌──────────────── stnav_gui env (PySide6) ───────────────────────────┐
│ GUI thread: MainWindow (화면, 버튼, 표, 레이어 렌더링)                 │
│  ├─ CameraWorker(QThread): Image/Video/V4L2 → Frame (backpressure)  │
│  ├─ AnalysisWorker(QThread): AnalysisRun (상태 머신)                 │
│  ├─ FunctionWorker(QThread): 모델 준비, replay 로드, 그림 내보내기    │
│  └─ GpuMonitor(QThread): nvidia-smi 실측                            │
│        │  JSON lines (stdin/stdout), 이미지는 run 폴더 파일 경로로 전달 │
└────────┼───────────────────────────────────────────────────────────┘
         ▼
┌──────── qwen_vl env (기존, 변경 없음) ───────┐
│ model_worker.py  (python -I, HF_HUB_OFFLINE=1)│
│  Qwen3-VL-4B bf16  ⇄  Grounding DINO tiny     │
│  (GPU 배타 상주: 필요할 때 교체)               │
└───────────────────────────────────────────────┘
```

- **Qt 충돌 회피**: GUI 환경은 `opencv-python-headless`를 쓴다. Qt를 번들한 OpenCV와 PySide6를 같은 프로세스에 두지 않는다.
- **worker 실행 방식**: `-I`(격리 모드)로 실행하고 `PYTHONPATH`를 제거한다. ROS 경로와 형제 모듈이 섞이지 않게 하기 위해서다.
- **중지**: 오케스트레이터가 취소 플래그를 세우고, 추론 중이면 worker 프로세스를 종료한다.

## 2. Agent = 고정 도구 목록 위의 상태 머신

`agent/orchestrator.py`는 작업 유형별로 정해진 단계를 실행한다.

```text
transport_and_place: parse_task → analyze_scene → detect_objects → inspect_target → build_scene_graph
                     → build_navigation_map → infer_destination → assess_placement → select_pickup_pose
                     → select_dropoff_pose → plan_path → validate_path → create_handoff_request → save_results
navigate_to_region : parse_task → analyze_scene → detect_objects → build_scene_graph → build_navigation_map
                     → infer_destination → select_dropoff_pose → plan_path → validate_path
                     → create_handoff_request → save_results
```

**모델이 할 수 있는 일과 없는 일**
- LLM/VLM은 다음 네 가지만 한다: 명령 해석, 장면 설명, 후보 크롭 검증(yes/no/uncertain), 하역 후보별 타당성 판정.
- 도구 선택, 단계 순서, 좌표 생성, 코드 실행은 하지 않는다.
- `actions`는 고정 템플릿에서 채운다. 모델 출력에 다른 action이 있어도 무시한다(테스트로 확인).

**단계 실패 처리**
- 실패한 단계는 `FAILED` 또는 `UNCERTAIN`으로 기록하고 원인 코드를 남긴다.
- 그 결과가 필요한 단계는 `SKIPPED (blocked: missing …)`가 된다.
- `save_results`는 항상 실행된다.
- 예기치 못한 예외는 traceback을 결과에 기록한 뒤 다시 발생시킨다. 숨기지 않는다.

## 3. 데이터 계약 (`contracts.py`)

- 다음 구조는 JSON-Schema 부분집합 검증기로 검사한다.
  - Task, Detection, Relation, Region, GoalPose, Path, Handoff, Result
- LLM/VLM 응답은 엄격한 JSON 파싱과 스키마 검증을 거친다. 실패하면 해당 단계를 실패로 처리하고 원문을 보존한다.
- 출처 표기(`knowledge`)는 네 가지다.
  - `observed_by_model`: 모델이나 알고리즘이 이 프레임에서 낸 결과
  - `inferred`: VLM의 의미 추론. 검증되지 않음
  - `manual_prior`: 사람이 입력한 지도나 시작점
  - `manual_test_data`: IMAGE_DEMO용 수동 데이터

## 4. Perception

**1) 후보 생성**
- Grounding DINO를 클래스별 프롬프트로 실행한다.
- 낮은 임계값(0.10)을 쓰고, 이미지 25% 이상 크기의 박스는 버리고, NMS를 적용한다.

**2) 검증**
- 상위 K개 후보를 크롭한다. 크롭에는 녹색 박스와 주변 여백을 넣고 확대한다.
- Qwen3-VL이 yes/no/uncertain으로 답한다. **yes만 채택한다.**
- 채택된 후보가 없으면, 설정된 클래스에 한해 Qwen grounding fallback을 쓴다. fallback 결과도 같은 검증을 거친다.

**3) 빨간 큐브 검사 (`target_inspection.py`)**
- 검증된 컨베이어 주변에 ROI를 잡는다. 여백은 등방성이다. 컨베이어가 없으면 전체 이미지를 쓴다.
- ROI를 균일 배율로 확대한다.
- 확대한 ROI에서 다음 후보를 모은다.
  - HSV 색상·형태 후보
  - DINO 후보(ROI 안, 전체 이미지 둘 다)
- 후보를 병합한다. IoU 또는 포함관계로 판단하고, 색상 blob이 있으면 그 박스를 대표 박스로 쓴다.
- 우선순위대로 Qwen 검증을 한다.
- 결정 규칙
  - yes가 1개면 선택한다.
  - 여러 개면 명령의 출발 기준(컨베이어)과의 `near` 관계로 좁힌다.
  - 그래도 여러 개면 `TARGET_AMBIGUOUS`다.
- **색상만으로는 절대 채택하지 않는다.**

**4) Scene graph**
- `near`, `overlaps_in_image`: 이미지 픽셀 박스 간격으로 계산한다. 상태는 `COMPUTED`이고 metric 아님으로 표시한다.
- `produced_by`처럼 정지 영상으로 증명할 수 없는 관계는 `UNVERIFIED`로 둔다. 관찰 가능한 근거(간격 등)만 첨부한다.

## 5. 목적지 체인 (v0.2)

v0.1의 설계(접근성을 의미 판단에 섞음)와 그 실패 분석은 [ANALYSIS_v0.1_destination_failure.md](ANALYSIS_v0.1_destination_failure.md)에 있다.

```text
① 의미 (semantics/region_grounding.py) ─ 로봇 지도·정답·접근성은 모델에 주지 않는다
  truck_object (검증된 박스)
   → 후보: geometric_bands (v0.1 기준선, 네 변의 띠)
         + grounded_structure (Qwen grounding "parking space of the truck" 등의 박스.
           트럭 포함률 ≥ 0.5 또는 인접, 장면 크기 아님, 트럭 박스 자체 아님, 중복 아님)
   → 후보마다 크롭 1장(흰 T = 트럭, 녹색 = 후보) → VLM yes / no / uncertain
   → semantic_regions (= yes인 후보, unverified)
     상태: SEMANTIC_REGION_FOUND | SEMANTIC_REGIONS_MULTIPLE | SEMANTIC_REGION_UNCERTAIN | NO_LOADING_REGION_FOUND
② 배치 (semantics/placement.py)
  object_placement_region = 의미 영역 − 검출 물체 박스(트럭 등) − 지도 장애물·금지 영역, 대상 크기 ½ 침식
   → PLACEMENT_REGION_FOUND | PLACEMENT_REGION_EMPTY   (표면과 그리퍼 도달은 미측정)
③ 접근 (navigation/goal_selector.py)
  robot_approach_region = 충돌 없는 로봇 중심 셀 ∩ 배치 영역에서 reach 이내 ∩ 원점(픽업 위치)에서 도달 가능
   → ROBOT_APPROACH_NOT_FOUND | ROBOT_APPROACH_UNREACHABLE | GOAL_SELECTED
  selected_robot_goal = 접근 영역에서 (경로 비용 + 1.5 × 거리)가 최소인 셀
  여러 의미 영역이 실행 가능하면 점수 최소인 것을 고르고 규칙을 기록한다(dropoff_choice).
④ 경로: A* + 직선화 + 픽셀 단위 검증
```

**결과 기록**
- `result.json`의 `destination`(및 `analysis/destination_chain.json`)에 다음을 담는다.
  - `semantic_loading_region`, `object_placement_region`, `robot_approach_region`, `selected_robot_goal`
  - 후보별 단계 표 `chain`
  - `failed_stage`와 `final_stage_status`
  - `goal_reach_check`: 목표와 앵커 사이 거리, reach 여유, 필요한 최소 reach
  - `claims`
- reach는 실측값(`handoff_geometry.gripper_reach_px`)이 없으면 `PROVISIONAL`이다(`goal_selection.*.max_dist_px` 개발 가정).
- `robot_task_success`는 항상 `NOT_CLAIMABLE`이다.

**명령 유형별 차이**
- `navigate_to_region`에는 배치 단계가 없다. 접근 앵커가 의미 영역 자체다.
- IMAGE_DEMO는 의미 단계만 수동 영역으로 바꾸고, ②–④는 같다.

**사전 zone 사용 원칙**
- `validated_loading_region`은 확정된(confirmed) 사전 zone과 IoU가 일치할 때만 채운다.
- 현재 Z06은 미확정이라 일관성 수치만 기록한다.

## 6. 지도와 경로 (IMAGE_SPACE_DEMO)

**지도 (`map_builder.py`)**
- 레이어를 분리한다: 통행 가능, 고정 장애물, 금지 영역, 동적 장애물(이번 프레임 검출), 의미 영역.
- 통행 가능 다각형 밖은 모두 막힌 것으로 본다. 바닥 색은 쓰지 않는다.
- 로봇 footprint는 반경 `robot_radius_px`인 원이다(가정값).
- 셀 중심에서 픽셀 거리장으로 잰 여유가 반경 이상이면 유효 셀이다.

**카메라 정렬 검사**
- 지도의 기준 이미지와 현재 프레임을 위상상관으로 비교한다.
- 이동 3 px 이하, 응답 0.3 이상이 아니면 `MAP_CAMERA_MISALIGNED`로 계획을 거부한다.

**접근 위치 (`goal_selector.py`)**
- 대상 다각형에서 [min, max] 거리 안에 있는 유효 셀을 후보로 잡는다.
- 시작점에서 Dijkstra 비용장을 계산해 도달 가능한 셀만 남긴다.
- 점수(경로 비용 + 1.5 × 거리)가 가장 낮은 셀을 고른다.
- 상태: `NO_ACCESSIBLE_CANDIDATE`, `CANDIDATES_UNREACHABLE`
- 객체 중심을 목표로 쓰지 않는다.

**경로 (`planner.py`)**
- 기존 `grid_astar.py`의 SAFETY_COST_ASTAR를 이식했다(8-연결, corner-cut 금지, 여유 기반 위험 비용).
- 그 뒤 line-of-sight 직선화를 한다.
- 경로가 없으면 빈 경로를 반환한다. 직선으로 대체하지 않는다.
- Theta*와 곡선 보간은 쓰지 않는다.

**검증 (`path_validator.py`)**
- 플래너 격자와 독립적으로 검사한다.
- 연속 polyline을 2 px 간격으로 샘플링한다.
- 각 점의 픽셀 수준 여유가 `반경 − 허용오차(격자 크기)` 이상이어야 한다.
- 시작점과 목표점이 일치하는지도 확인한다.
- 위반하면 경로 상태가 `REJECTED_BY_VALIDATOR`가 된다.

## 7. 인계 (`robot/handoff.py`)

- 요청 형식은 `PICKUP_HANDOFF`, `PLACEMENT_HANDOFF`, `NAVIGATION_GOAL`이다.
  - 각 요청에 대상 ID, 클래스, 색, 이미지 좌표 접근 자세, `approach_pose_metric=null`, `hardware_executable=false`가 들어간다.
- 상태 머신은 다음 순서로만 움직인다.

```text
WAITING_FOR_PICKUP_CONFIRMATION
  → (사용자 시뮬레이션 확인 또는 외부 시뮬레이터) → WAITING_FOR_PLACEMENT_CONFIRMATION
  → COMPLETED_SIMULATED
```

- 다른 출처의 확인은 거부한다.
- `send_to_hardware()`는 항상 예외를 낸다.
- ROS2 연동은 인터페이스(JSON 요청 파일)만 준비했다. 실제 publish는 구현하지 않았다.

## 8. 재현성

- **실행 폴더**: 매 실행마다 새로 만들고 기존 파일은 덮어쓰지 않는다. 인계 상태는 `state_NNN.json`으로 추가한다.
- **manifest**: 입력과 디코딩 픽셀의 SHA-256, 코드 해시(src와 run.py), 모델 commit, 프롬프트 sha, 설정 스냅샷, 파일 해시를 기록한다.
- **replay**: 저장된 결과와 지도 스펙으로 지도를 다시 만들어 레이어 해시를 대조한다. 렌더링도 다시 해서 그림이 픽셀 단위로 같은지 확인한다.

## 9. 개발 중 변경 기록 (참고 이미지 = 개발 데이터)

| 변경 | 이유(실측) |
|---|---|
| GPU 배타 상주 | 4B와 DINO를 동시에 올리면 CUDA OOM |
| DINO 임계값 0.20 → 0.10, `roller conveyor` 프롬프트 추가 | 0.20에서는 실제 컨베이어가 후보에 없었음 |
| 컨베이어 검증 설명을 "production line containing a conveyor"로 확대 | Qwen이 컨베이어와 로봇팔이 함께 든 박스를 "robot arm"으로 답함 |
| ROI 여백을 등방성으로 변경 | 축별 여백 때문에 큐브가 ROI 경계에서 잘림 |
| 큐브 후보 병합에 포함관계 추가 | 정확한 HSV 박스가 큰 DINO 박스와 IoU 0.10이라 병합되지 않아 검증 순위 밖으로 밀림 |
| 접근 위치 거리 가중치 0.5 → 1.5, 픽업 도달 거리 180 → 140 px | 픽업 위치가 도달 범위 가장자리에 선택됨 |
| 하역 영역 v1(단일 선택) → v2(후보별 타당성 + 접근성) | v1이 로봇 작업 공간 밖 영역을 선택함. v2에서도 VLM 판단은 같았고, 시스템은 INACCESSIBLE로 기록함 |
| v0.2: 의미 단계에서 접근성 제거, 후보별 단일 판정, 구조 grounding 후보 추가, 배치·접근 단계 분리 | v0.1이 매트 밖 영역을 선택하고 의미 영역도 남기지 않음. 진단에서 VLM이 트럭 방향을 틀리게 인식. Qwen grounding은 주차 구획을 찾음 (분석 문서 참고) |

이 변경들은 모두 같은 개발 이미지에서 이루어졌다. 성능 평가에는 새로 촬영한 이미지(test split)를 써야 한다.

v0.2 변경도 같은 개발 이미지에서 이루어졌다. v0.1 동결 스냅샷은 `archive/v0.1_2026-11-15/`에 있다.
