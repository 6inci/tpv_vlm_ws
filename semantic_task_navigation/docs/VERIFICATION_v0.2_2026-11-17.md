# 검증 기록 — v0.2 (2026-11-17)

- 참고 이미지는 **개발 데이터**다. 여러 번 조정에 사용했으므로 아래 비교는 개발 장면의 동작 비교이며 성능 평가가 아니다.
- 좌표와 거리는 모두 이미지 픽셀이다.

## 1. 같은 개발 장면에서 v0.1과 v0.2 비교 (실제 Qwen3-VL-4B + Grounding DINO)

| 항목 | v0.1 `20261115_225944_49f5` | v0.2 ablation (기하 띠만) `20261117_111211_1210` | v0.2 `20261117_111627_d4cf` |
|---|---|---|---|
| 코드 해시 | `b7ba777e…` | `c85d64a5…` | `254cfb31…` |
| 후보 | 기하 띠 4 | 기하 띠 4 | 기하 띠 4 + 구조 1 ("parking space of the truck") |
| VLM 판정 방식 | 4개를 한 장에 겹쳐 선택 + 접근성 | 후보별 단일 판정 | 후보별 단일 판정 |
| 판정 결과 | 1 no / **2 yes(위쪽, 매트 밖)** / 3 no / 4 no | 1–4 모두 no | 1–4 no / **5 yes(구획)** |
| 의미 단계 | `SEMANTIC_REGION_INACCESSIBLE` (의미 영역 기록 없음) | `NO_LOADING_REGION_FOUND` | `SEMANTIC_REGION_FOUND` |
| Z06 초안과 IoU (평가용, 미확정) | – | – | 0.722 |
| 배치 영역 | – | – | `PLACEMENT_REGION_FOUND` 4,729 px (구획 39,624 → 트럭 제외 13,224 → 8 px 침식) |
| 로봇 접근 영역 | – | – | 충돌 없음 128셀 = 도달 가능 128셀 (reach 180 px **PROVISIONAL**) |
| 선택 목표 | – | – | (1316, 372) px, 배치 영역까지 177.5 px, **여유 2.5 px** |
| 경로 B | 없음 | 없음 | 696 px, 검증 통과(최소 여유 36 px) |
| 최종 상태 | `PARTIAL_PLAN` | `PARTIAL_PLAN` | `PLAN_COMPLETE`, `DESTINATION_PLANNED` |
| 로봇 작업 성공 주장 | 불가 | 불가 | **불가** (`NOT_CLAIMABLE`) |

**반복성**
- v0.2를 3번 실행했다: `111047_aa91`, `111321_c086`, `111627_d4cf`.
- 의미 판정, 배치 영역, 목표, 경로가 모두 같았다(`do_sample=false`).
- 최종 실행만 `goal_reach_check` 필드가 추가된 코드로 실행했다.

**해석**
- **판정 방식 변경**: 후보를 하나씩 판정하자, v0.1의 잘못된 "위쪽" 선택이 사라졌다.
  - 대신 기하 띠만으로는 하역장을 찾지 못했다(ablation).
- **구조 후보 추가**: 모델이 찾은 주차 구획 후보가 "yes"로 판정되어 체인이 이어졌다.
  - 이 구획은 사람 초안 Z06과 IoU 0.72다. Z06은 미확정 초안이고 모델 입력에 쓰지 않았다.
- **도달 거리 의존**: 이 결과는 **도달 거리 가정에 전적으로 의존한다.** 아래 §2를 참고한다.

## 2. 도달 거리 민감도 (v0.2 결과의 배치 영역 기준)

| reach (px) | 100 | 120 | 140 | 145 | 150 | 160 | 180 | 220 |
|---|---|---|---|---|---|---|---|---|
| 접근 셀 (충돌 없음 = 도달 가능) | 0 | 0 | 0 | 0 | 0 | 3 | 128 | 388 |
| 결과 | NOT_FOUND | NOT_FOUND | NOT_FOUND | NOT_FOUND | NOT_FOUND | FOUND | FOUND | FOUND |

- 배치 영역에서 가장 가까운 충돌 없는 로봇 위치까지는 **155.2 px**다(`nearest_collision_free_cell_px`).
- 실제 그리퍼와 온보드 카메라가 약 156 px(이미지) 이상을 커버하지 못하면 이 장면의 하역은 불가능하다(`ROBOT_APPROACH_NOT_FOUND`).
- 로봇 통행 영역(Z02)과 트럭 구획 사이에 연결 통로가 없다는 점은 지도 초안에도 적혀 있다.

## 3. 테스트

| 묶음 | 결과 |
|---|---|
| v0.2 전체 (`STNAV_RUN_MODEL_TESTS=1 python -m pytest -q`) | **80 passed** (실제 모델 3개 포함) |
| v0.1 동결 스냅샷 (`archive/v0.1_2026-11-15`, 실제 모델 포함) | **68 passed**, 파일 해시 일치 |
| v0.1 실행 replay (v0.2 코드) | `225944_49f5`, `224618_0e4d`, `225333_ce4c`: 그림 10개 IDENTICAL (11번은 실행에 없음) |
| v0.2 실행 replay | 그림 11개 IDENTICAL |

### 새 조건 테스트 (`tests/test_destination_chain.py`)

이 조건들은 실제 촬영 데이터가 없다. **합성 장면이나 가짜 모델 응답으로만 검증했다.**

| 조건 | 테스트 | 기대 결과 |
|---|---|---|
| 적절한 하역장과 접근 위치 존재 | `test_e2e_region_and_approach_exist` (IMAGE_DEMO 수동 영역) | `DESTINATION_PLANNED`, reach PROVISIONAL, 성공 주장 불가 |
| 하역장은 있으나 로봇 접근 불가 | `test_e2e_region_exists_but_robot_cannot_approach`, `test_region_without_robot_approach_is_its_own_failure` | 배치 FOUND, `ROBOT_APPROACH_NOT_FOUND`, 경로 B 없음 |
| 접근 셀은 있으나 도달 불가 | `test_approach_exists_but_unreachable` | `ROBOT_APPROACH_UNREACHABLE` |
| 트럭은 있으나 유효한 하역장 없음 | `test_truck_present_but_no_valid_loading_region`, `test_e2e_no_loading_region` | `NO_LOADING_REGION_FOUND` / 의미 단계 실패 |
| 하역장 영역이 트럭 자체뿐 | `test_placement_empty_when_region_is_the_truck_itself`, `test_e2e_region_is_only_the_truck` | `PLACEMENT_REGION_EMPTY` |
| 하역장 후보 여러 개 | `test_multiple_plausible_regions_defer_the_choice`, `test_multiple_feasible_regions_choose_lowest_score` | 의미 단계 `MULTIPLE`, 이후 점수 최소 선택과 규칙 기록 |
| 트럭 위치 변경 | `test_candidates_follow_a_moved_truck` | 후보가 트럭을 따라 이동 |
| 모델 입력 격리 | `test_one_judgement_per_candidate_and_no_map_in_model_input` | 후보마다 이미지 1장, 프롬프트에 지도·정답 없음 |
| 구조 후보 검증 | `test_structure_candidates_are_checked_against_the_truck` | 장면 전체, 트럭 자체, 무관 박스, 중복 박스 거부 |
| 사람 정답 격리 | `tests/test_gui.py::test_human_reference_regions_are_display_only` | 정답 좌표가 실행 기록에 없음, 별도 레이어로 표시 |

## 4. 해결된 것과 남은 것

**해결됨**
- 의미 판단과 실행 가능성을 섞던 문제를 단계 분리로 해결했다.
  - "의미적으로 부적합", "놓을 곳 없음", "접근 불가", "도달 불가", "경로 없음"을 서로 다른 코드로 기록한다.
- VLM이 고른 의미 영역이 실패 시에도 결과에 남는다.
- 접근 영역 정의에 원점 도달 가능성을 포함했다. 필요한 최소 도달 거리와 여유도 기록한다.
- 사람 정답 영역을 GUI에서 입력, 저장, 표시할 수 있다. 분석 입력에는 들어가지 않는다.

**남음**
- **물리량 미측정**
  - 그리퍼 도달 범위, 온보드 카메라 시야, 배치 표면이 측정되지 않았다.
  - 그래서 v0.2의 `PLAN_COMPLETE`는 가정에 의존하는 이미지 좌표 계획일 뿐이다.
  - 선택 목표는 가정 한계에서 2.5 px 여유밖에 없다.
- **목표 선택 기준**
  - 하역 목표 점수가 경로 비용을 함께 반영해서, 배치 영역에서 먼 쪽(177.5 px)을 골랐다. v0.1과 같은 규칙이다.
  - 실측 도달 거리가 정해지면 "배치 영역과의 거리 우선" 정책이 필요한지 다시 판단해야 한다.
- **판정의 일반화**
  - 구조 질의 문구와 판정 프롬프트는 개발 이미지에서 정했다. 새 장면에서 일반화되는지는 모른다.
  - 같은 모델이 트럭 방향을 틀리게 인식한 사례가 있다(분석 §B).
- **사람 정답 없음**
  - 하역장과 배치 영역의 사람 정답이 아직 없다. 정확도 판정은 보류한다(`NO_GT (pending)`).
- **지도 미검토**
  - 지도(Z02 등)는 `DRAFT_UNCONFIRMED` 상태다.
