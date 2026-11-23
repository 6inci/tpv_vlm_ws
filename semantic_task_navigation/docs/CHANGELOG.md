# CHANGELOG

## v0.2 — 2026-11-17: 의미 영역과 로봇 작업 목표 분리

- **동결 기준선**: `archive/v0.1_2026-11-15/`
  - v0.1 코드·설정·테스트를 수정 전에 복사했다. 코드 해시는 `b7ba777e…`로 대표 실행과 같다.
  - 독립 실행된다. 테스트 68개 통과, 파일 해시(`SHA256SUMS`) 일치.
- **보존**: 기존 실행 결과와 지도, 데모 데이터, 환경(conda, CUDA, ROS2)은 바꾸지 않았다.
- **실패 원인 분석**: [ANALYSIS_v0.1_destination_failure.md](ANALYSIS_v0.1_destination_failure.md)

### 데이터 흐름 변경

```text
v0.1: 후보(4 띠) → VLM 다중 후보 선택 + 접근성 → 영역 1개 또는 실패(INACCESSIBLE) → 접근 위치 → 경로
v0.2: 후보(4 띠 + 구조 grounding) → 후보별 VLM 판정(의미만)        → semantic_loading_region(s)
      → assess_placement (배치 가능 영역, 기하)                     → object_placement_region
      → select_dropoff_pose (도달 거리 안 AND 도달 가능 셀, 점수 최소) → robot_approach_region, selected_robot_goal
      → plan_path / validate_path
```

### 단계별 실패 코드 (서로 다른 실패로 기록)

| 단계 | 코드 |
|---|---|
| 의미 | `NO_LOADING_REGION_CANDIDATES`, `NO_LOADING_REGION_FOUND`(트럭은 있으나 모든 후보가 no), `SEMANTIC_REGION_UNCERTAIN`, `SEMANTIC_REGIONS_MULTIPLE`(실패 아님, 선택을 뒤 단계로 미룸) |
| 배치 | `PLACEMENT_REGION_EMPTY` |
| 접근 | `ROBOT_APPROACH_NOT_FOUND`(도달 거리 안에 충돌 없는 셀 없음), `ROBOT_APPROACH_UNREACHABLE`(셀은 있으나 원점과 연결되지 않음) |
| 경로 | `NO_PATH`, `REJECTED_BY_VALIDATOR` |

### 파일별 변경과 이유

| 파일 | 변경 | 이유 |
|---|---|---|
| `semantics/region_grounding.py` | 의미 단계에서 접근성을 제거. 구조 후보(`structure_candidates`)와 트럭 연관성 검사 추가. 후보별 단일 판정(`judge_candidates`). 상태를 FOUND / MULTIPLE / UNCERTAIN / NO_LOADING_REGION_FOUND로 세분화. v0.1 다중 후보 선택 코드는 삭제(스냅샷에 보존) | 분석 A, B, C. 의미 판단과 실행 가능성을 섞지 않음 |
| `semantics/placement.py` (신규) | 의미 영역에서 검출 물체(트럭 등)·지도 장애물·금지 영역을 빼고 대상 크기 절반만큼 침식. 면적별 이유 기록 | "하역장은 있으나 놓을 곳이 없음"을 별도 실패로 구분 |
| `navigation/goal_selector.py` | 다중 다각형 앵커 지원. `reach` 근거(PROVISIONAL / CONFIGURED) 기록. `nearest_valid_cell_to_anchor_px` 추가. v0.1 `accessible_cell_count` 삭제 | 분석 E. 접근 영역 = 도달 거리 안 AND 도달 가능. 필요한 최소 도달 거리를 기록 |
| `agent/orchestrator.py` | `assess_placement` 단계 추가. `select_dropoff_pose`가 실행 가능한 후보 중 점수 최소를 선택하고 선택 규칙을 기록. 결과에 `destination` 블록 추가(네 계층, 후보별 단계 표, `goal_reach_check`, claims). `analysis/placement_regions.json`, `analysis/destination_chain.json` 저장 | 요구사항 4·7: 단계 분리와 단계별 실패 이유 |
| `visualization.py` | 레이어 `placement_region`, `approach_region`, `reference_regions` 추가. 화면 `DESTINATION_CHAIN`, 그림 `11_destination_chain` 추가. 기존 레이어 동작은 그대로 | 요구사항 5. v0.1 실행이 같은 그림으로 replay됨(테스트로 확인) |
| `gui/main_window.py` | 지도 편집기에 `정답: 하역장(사람)`, `정답: 배치영역(사람)` 유형 추가. `정답 저장`·`정답 불러오기`(`data/reference/*.json`, 덮어쓰기 금지, 작성자 입력). 표시 전용 레이어 | 사람 정답을 모델 출력과 구분해 표시. 분석 입력에는 넘기지 않음(테스트로 확인) |
| `config/planning.yaml` | `destination`, `placement`, `handoff_geometry`(gripper_reach_px, onboard_camera_view_px = null) 추가 | 실측되지 않은 값을 확정하지 않음 |
| `config/models.yaml` | `prompts.region_judge` (v1, 후보 1개 판정) 추가 | 분석 B |
| `scripts/evaluate_run.py` | 사람 정답의 `placement_region_polygons`와 비교(배치 IoU, 목표와 정답 배치 영역 거리). 정답이 없으면 `NO_GT (pending)` | 정답 판정 보류 지원 |
| `run.py` | `--set SECTION.KEY=JSON` 설정 덮어쓰기(manifest의 `config_overrides`에 기록). headless replay가 실행에 없는 그림은 `NOT_IN_RUN`으로 표시 | 같은 장면에서 ablation 비교. v0.1 실행 replay |
| `tests/test_destination_chain.py` (신규, 17개) | 합성 데이터와 가짜 모델 응답으로 조건별 검증. v0.1 실행 replay 회귀 포함 | 요구사항 8 |
| `tests/test_gui.py` | 정답 영역이 실행 기록에 들어가지 않는지 확인하는 테스트 추가 | 요구사항 5 |
| `tests/test_semantics_handoff_calibration.py` | v0.1 `test_region_selection` 제거. v0.1 동작은 스냅샷 테스트가 계속 검증 | 의도적인 동작 변경 |
