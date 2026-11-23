# 의미 맥락 지도 + 이동 비용 지도 + 작업 인지 의사결정 통합 (E1, 2026-11-19)

- 입력: 명령 "컨베이어에서 생산된 빨간 큐브를 출하할 수 있는 적절한 장소로 옮겨줘." + E1 이미지 `undistorted_25070552942982.png` (sha256 `6535b84a…`).
- 근거 실행: `results/20261119_111050_semnav/`. 개발 중 실패하거나 수정 전 실행 4개는 `superseded_runs/`에 보존했다.
- **모델 호출 0회**, GPU·하드웨어·ROS2 사용 없음. v0.2 코드 지문 `254cfb31da7f…`는 실행 전후 같고, 새 폴더 밖에서 수정된 파일은 없다.
- 좌표는 잠정 바닥 변환(`PROVISIONAL_UNVALIDATED`, 맞춤 RMS 3.57 cm, 독립 이음매 편차 ≤ 3.5 cm)과 DRAFT 지도 기준이다. 수치는 그 모델 안에서만 정확하다.
- 정답이 없다. 8장의 큐브 라벨은 `AI_PRELIMINARY_REVIEW`이며 결정 입력에 넣지 않았고 정확도·성공률은 계산하지 않았다.

## 0. 결론

1. **세 실험 모두 구현·실행했고, 하나의 명령과 이미지에서 한 번에 재현된다.** 출발 → 경로 A → `VERIFY_REQUIRED` → `PICKUP_ASSUMED` → 경로 B → 배치 접근 → `PLACEMENT_ASSUMED`가 이어진다.
2. **이동 비용 지도는 v0.2 A*와 정확히 일치한다.** 같은 지도·차체·간선 비용에서 무작위 도달 셀 40개의 비용 차이가 0이다. 도달 가능한 셀 1,353개 모두 독립 래스터 충돌 검사를 통과했고, 이전 실험의 경로 A(83, 113 cm, 236.3 cm)도 그대로 재현된다.
3. **목표 선택은 의미 맥락의 등급 규칙에 달려 있다.** 시각 근거만으로는 PROBABLE 후보 3개가 동률이라 `AMBIGUOUS`다. 의미 맥락의 "컨베이어 박스 안(on) > 근처(near)" 규칙을 쓰면 `cand_r1`이 선택된다. on과 near를 같게 보면 다시 `AMBIGUOUS`다. 이 규칙은 E1 후보 위치를 이미 본 뒤 정한 개발 규칙이다.
4. **이동 비용은 이 사진에서 선택을 바꾸지 않았다.** 상위 후보 셋 모두 경로 A·B가 있어서다. 대신 동률일 때 확인 순서를 정했다(간격 13.7 → 68.9 → 72.9 cm). 이동 비용만 쓰는 기준선 D는 Qwen 답이 없는 후보 `cand_r9`(로봇 바로 옆)와 이미지 전체 상자 `area_1`을 골랐다. "가까운 것 고르기"가 의미와 무관한 대상을 고른다는 사례다.
5. **물리적 실행 가능성은 미검증이다.** 픽업 간격 13.7 cm와 배치 간격 38.0 cm를 그리퍼가 메울 수 있는지 모른다. 목표와 하역 구획은 모두 주행 영역 밖이다.

## 1. 구현·실행 상태

| 항목 | 상태 | 근거 |
|---|---|---|
| Exp 1 의미 맥락 지도 | IMPLEMENTED, EXECUTED | 노드 60, 관계 51 |
| Exp 1 구조 규칙(근거 없는 관계 금지, 명령·VLM 관계는 SUPPORTED 불가) | VALIDATED | `validation.json` experiment_1, 단위 테스트 |
| Exp 1 관계·의미 추론의 정확도 | UNVERIFIED | 정답 없음 |
| Exp 2 이동 비용 지도 | IMPLEMENTED, EXECUTED | 0°, 90°, 회전 경유 90° |
| Exp 2 A* 일관성·충돌 | VALIDATED (모델 내부) | 차이 0.0, 위반 0 |
| Exp 2 실제 치수 기반 안전성 | UNVERIFIED | 잠정 변환, DRAFT 지도 |
| 90° 방향으로 출발 | FAILED | 출발점에서 90° 차체가 들어가지 않음 |
| Exp 3 결정 모듈과 A/B/C(+D) 비교 | IMPLEMENTED, EXECUTED | `ablation_results.csv` |
| Exp 3 선택 결과(`cand_r1`, `area_5`) | CONDITIONAL | `VERIFY_REQUIRED`, 정답 없음 |
| 경로 A·B 통합 | EXECUTED, VALIDATED (모델 내부), CONDITIONAL | 전신 검사·v0.2 검증·층별 겹침 통과, A·B 연결 |
| 픽업·배치 가능성 | UNVERIFIED | 그리퍼 도달 거리 미측정 |
| 8장 보조 이미지 | 사용 안 함 | E1 외 사진에는 DINO·Qwen 결과와 바닥 변환이 없다 |

## 2. 재사용한 기존 기능

| 기존 기능 | 사용처 |
|---|---|
| `task_route_integration.build_context` | 잠정 바닥 변환, DRAFT 지도, 저장된 로봇 시작점, 트럭·컨베이어·콘 박스, Qwen L2 응답, 하역 구획과 빈 바닥 |
| `task_route_integration.summarize / swept_layer_overlap / layers / draw_path` | 경로 기록, 층별 겹침 검사, 그림 |
| `task_plan_prototype.plan_variant / approach_goal / make_leg / RectNavMap / body_free / body_gap_cm` | 직사각형 차체(35×24 + 3.5 cm/변), 접근점 규칙, A*, 전신 검사, 간격 |
| v0.2 `cost_field`, `astar`, `plan`, `validate_path`, `_distance_to_polygon`, `rule_based_parse`, `box_gap`, `containment`, `relations.near_px = 150`, `contain_ratio = 0.8` | 비용 지도, 일관성 검사, 명령 파싱, 2D 관계 임계값 |
| `candidate_retention_policy` 결과 | 후보 38개와 상태 |
| `qwen_crop_comparison` 결과 | 후보 4개의 Qwen 보기 A/B/C |

`AnalysisRun`은 만들지 않았다(모델 호출 중심 상태기계). 상태 이름은 `task_route_integration.PLAN_STATES`를 따랐다.

## 3. 새로 구현한 기능 (모두 이 폴더 안)

| 파일 | 내용 |
|---|---|
| `context_map.py` | 작업 중심 노드, 관계 생성 규칙, 출처(`INSTRUCTION_GIVEN` / `VLM_INFERRED` / `GEOMETRY_DERIVED`)와 상태(`SUPPORTED` / `UNVERIFIED` / `UNKNOWN`), VLM 언급·충돌 기록 |
| `cost_map.py` | 셀 분류 7종, 회전 가능 셀에서만 방향을 바꾸는 다중 시작 Dijkstra(v0.2와 같은 간선 함수), 비용 지도 기반 접근점, A* 일관성·충돌 검사 |
| `decision.py` | 사전식 근거 단계 정책: 결정 요인에서 엄격히 앞설 때만 `SELECTED`, 동률은 `AMBIGUOUS`, 상위 단계가 모두 주행 불가면 `DEFERRED_NAVIGATION` |
| `run_integration.py` | 입력 수집, 세 Phase 실행, 목적지 평가, 비교표, 그림, 검증, manifest |
| `test_semnav_offline.py` | 합성 입력 테스트 10건 |

## 4. Exp 1 — 의미 맥락 지도

**포함 규칙**: 명령의 대상 클래스(빨간 큐브 후보 38개 전부)와 출처 클래스(컨베이어), 로봇, 목적지 후보(Qwen 기능 영역)와 그 근거(하역 구획, 빈 바닥, 트럭), 움직임을 제한하는 지도 구역, 목적지 옆의 검증된 장애물(콘 6개). 그 밖의 Qwen 언급(로봇 팔, 화이트보드 등)은 텍스트로만 남겼다. `area_1`은 상자가 이미지 전체라 위치를 가리키지 못해 제외했다.

**관계 51개**: 기하 49, 명령 1, VLM 1. SUPPORTED 49, UNVERIFIED 2.

| 관계 | 출처 / 상태 | 내용 |
|---|---|---|
| task → `conveyor_1` `source_referenced_by_instruction` | 명령 / UNVERIFIED | "컨베이어에서 생산된". 생산 이력은 정지 영상으로 확인할 수 없다 |
| `area_5` → task `candidate_destination_for` | VLM / UNVERIFIED | 저장된 Qwen 응답(더 짧은 문장에 대한 것) |
| 후보 → `conveyor_1` `on` (6), `near` (8) | 기하(2D) / SUPPORTED | `cand_r1`은 컨베이어 박스 안(포함률 1.0). `cand_r3`, `cand_r6`는 박스 위쪽 61, 57 px. 3D로 "위에 놓임"은 확인하지 않았다 |
| `area_5` ↔ 트럭, 콘 4, 하역 구획, 빈 바닥 / `area_2` ↔ 컨베이어 / `area_6` ↔ 로봇 `associated_with` | 기하(2D) / SUPPORTED | VLM 영역의 위치를 관측 객체에 연결할 뿐, 영역의 기능은 여전히 VLM 추론이다 |
| 트럭 ↔ 하역 구획 `associated_with` | 기하 / SUPPORTED | 저장된 검사: 트럭 하단이 구획 안 0.905 (기준 0.9) |
| 콘 6 → 하역 구획 `near` | 기하(2D) / SUPPORTED | 73–116 px |
| `cand_r1` → Z07(컨베이어 작업셀, 진입 금지) `inside` 등 후보 13개 | 기하(바닥) / SUPPORTED | 바닥 투영 (51.5, 92.6) cm. 높이를 몰라 시차로 밀린 위치다 |
| Z03·Z04 ⊂ Z02, Z02–Z01(1.5 cm)·Z02–Z07(0.5 cm) 인접, 로봇 ⊂ Z02 | 기하(바닥) / SUPPORTED | 하역 구획과 Z02 사이는 31.8 cm 떨어져 있어 인접하지 않다 |

**보존한 충돌**: Qwen은 "빨간 큐브가 이미지에 보이지 않는다"고 했지만 탐지 후보는 38개(PROBABLE 3)다. Qwen은 콘 12개를 언급했지만 검증된 콘 박스는 6개다. 어느 쪽도 정답으로 취급하지 않았다.

**새 Qwen 호출을 하지 않은 이유**: 저장된 응답에 목적지, 기능 영역 상자, 로봇·컨베이어·트럭·콘·후보 4개의 크롭 판정이 이미 있다. 공간 관계는 탐지 박스와 바닥 좌표로 Python이 계산할 수 있다. 관계 추론을 VLM에 새로 맡길 필요가 없었다.

## 5. Exp 2 — 이동 비용 지도

| 항목 | 값 |
|---|---|
| 격자 | 2 cm 셀, 115 × 167. 이미지 px → 바닥 cm는 호모그래피로 변환한다(사진 전체에 단일 px/cm 비율을 쓰지 않음) |
| 차체 | 35×24 cm + 3.5 cm/변 = 42×31 cm. 사각형 전체가 자유 바닥 위에 있어야 하고(침식), 지도마다 방향은 고정이다 |
| 비용 | v0.2 `cost_field` (A*와 같은 간선, 8방향, 대각선 모서리 통과 금지, 위험 항 포함) + 거리 전용 측지 길이 |

| 셀 분류 (0°) | 셀 | 면적 cm² |
|---|---|---|
| REACHABLE | 1,353 | 5,412 |
| UNREACHABLE_FOOTPRINT (주행 구역이지만 차체가 안 들어감) | 4,725 | 18,900 |
| UNREACHABLE_DISCONNECTED | 0 | 0 |
| STATIC_OBSTACLE | 1,123 | 4,492 |
| NO_GO | 2,283 | 9,132 |
| UNKNOWN_UNVERIFIED (어느 주행 구역에도 없음, 주행 불가로 처리) | 7,106 | 28,424 |
| OUTSIDE_WORKSPACE | 2,615 | 10,460 |

- 도달 영역은 Z03·Z04를 둘러싼 고리 모양 통로다. 중심 기준 x 81–293 cm, y 81–161 cm. 최대 측지 길이는 452 cm, 최대 v0.2 비용은 642 cm 상당이다.
- **90°로 출발하는 경우는 실패다.** 출발점에서 y방향 42 cm 차체가 Z04(y ≈ 137)와 Z02 아래 경계(y ≈ 177) 사이 약 40 cm에 들어가지 않는다. 0° 지도는 "출발 방향 0°"를 가정한다(실제 방향은 모름).
- **방향 전환**: 출발점(여유 19.75 cm)과 픽업 접근점(22.75 cm) 모두 회전에 필요한 26.1 cm보다 작다. 회전 가능한 도달 셀은 173개(x 87–185, y 91–151 cm)이고, 그 셀을 거쳐 90°로 갈 수 있는 셀은 785개다. 회전 시간과 동역학은 모델링하지 않았다.
- **v0.2 비용과 경로 길이를 구분해야 한다.** `cand_r1` 접근점까지 v0.2 비용은 342.0 cm 상당(위험 항 포함), 격자 측지 길이는 241.1 cm, 단순화한 실제 경로는 236.3 cm다.

| 검증 | 결과 |
|---|---|
| 무작위 도달 셀 40개: v0.2 `astar()` 비용과 지도 값 비교 | 최대 차이 0.0 |
| 다중 시작 Dijkstra(출발점 하나)와 v0.2 `cost_field` 비교 | 같음 |
| 비용 지도 접근점과 `approach_goal` 비교 (후보 4개) | 같음 (목표 셀의 비용 = A* 비용, 예: 1367.877 px) |
| 도달 셀의 독립 충돌 검사 (cm 다각형 4 px/cm 래스터) | 0°: 1,353개 중 0, 회전 경유 90°: 785개 중 0 위반 |
| 이전 경로 A 재현 (`cand_r1`) | (83, 113) cm, 236.3 cm로 같음 |
| 대각선 모서리 통과 | 합성 지도 테스트로 확인 |
| 도달 불가 셀의 A* 재계획 | 0°에서는 해당 셀 없음 (끊긴 셀 0) |

`placement_reach_map`의 "차체–배치점 간격"과 이 지도의 "출발점에서의 경로 비용"은 다른 양이다.

## 6. Exp 3 — 후보 선택과 경로 통합

**정책** (`config.yaml`, 실행 전에 선언): 결정 요인을 사전식 순서로 비교한다. 시각 → 의미 → 주행 가능성(C만). 결정 요인에서 1위가 2위보다 엄격히 앞설 때만 `SELECTED`, 아니면 `AMBIGUOUS`다. 동률 해소 요인(간격, v0.2 점수, 탐지 점수)은 순서만 정하고 결정하지 않는다. 주행 가능성이 의미 요인 뒤에 있어서, 경로가 짧아도 근거가 약한 후보가 앞설 수 없다(단위 테스트로 확인).

| 조건 | 의미 등급 | 목표 | 목적지 |
|---|---|---|---|
| A 시각 | – | AMBIGUOUS {r1, r3, r6} (모두 PROBABLE) | NOT_DETERMINABLE |
| B 시각+의미 | 등급(on > near) | **SELECTED `cand_r1`** | SELECTED `area_5` (Qwen 선택 + 구획 근거) |
| C 시각+의미+이동 | 등급 | **SELECTED `cand_r1`**, 경로 A 236.3 + B 162.1 cm | SELECTED `area_5` → 구획 빈 바닥 |
| B 시각+의미 | 이진(on = near) | AMBIGUOUS {r1, r3, r6} | – |
| C 시각+의미+이동 | 이진 | AMBIGUOUS, 확인 순서 r1 → r6 → r3 (간격 13.7, 68.9, 72.9 cm) | – |
| D 이동 비용만 | – | SELECTED `cand_r9` (PENDING_VERIFICATION, Qwen 답 없음, 로봇 출발점 바로 옆, v0.2 점수 39.8) | SELECTED `area_1` (이미지 전체 상자, 경로 0 cm) |

- 3개 PROBABLE 후보 모두 경로 A·B가 생성·검증됐다. 그래서 이 사진에서는 이동 비용이 선택을 바꾸지 않았다.
- D가 고른 `cand_r9`는 출발점이 이미 접근 위치라 경로 A가 길이 0이다(`ZERO_LENGTH_A`). D가 2–4위로 고른 r38, r24, r8도 모두 로봇 근처의 PENDING 후보다.
- 목적지에서 Qwen 영역 상자 자체(`area_5`)로 가면 접근점은 (203, 81) cm, 간격 7.8 cm다. 반면 실제 빈 바닥까지는 38.0 cm가 남는다. VLM 영역 상자가 실제 배치면보다 주행 영역 쪽으로 넓다.
- 선택 결과가 AI 예비 라벨과 일치하는지는 비교표에 참고로만 적었다(B·C 등급: 일치, D: 불일치). 정답이 아니므로 성능 근거로 쓰지 않는다.

**통합 계획** (C, 등급, `integrated_task_plan.json`)

| 단계 | 내용 |
|---|---|
| 출발 | (293, 157) cm, ESTIMATED, 방향 0° 가정 |
| 경로 A | (293,157) → (105,155) → (101,153) → (83,113), 236.3 cm. 전신 검사·v0.2 검증·층별 겹침 통과 |
| VERIFY_REQUIRED | Qwen 보기 A/B/C = no/yes/no, Qwen L2는 "큐브가 안 보임", 라벨은 AI 예비 |
| PICKUP_ASSUMED | 그리퍼가 메워야 할 간격 13.7 cm (도달 거리 미측정) |
| 경로 B | (83,113) → (101,89) → (233,83), 162.1 cm, 0°. 같은 검사 통과, A 끝 = B 시작 |
| PLACEMENT_ASSUMED | `PLACEMENT_UNVERIFIED`, 빈 바닥까지 간격 38.0 cm |

**이전 배치 분석과의 차이**: `placement_reach_map`은 "간격이 가장 작은 배치점"을 기준으로 (281, 83) cm, 210.1 cm, 간격 36.8 cm를 얻었다. 이번에는 경로 A와 같은 접근 규칙(가장 가까운 도달 위치 + 6 cm 여유 안에서 경로 비용 최소)을 써서 (233, 83) cm가 나왔다. 주행은 48 cm 짧고 간격은 1.2 cm 크다. 이 위치에서 가장 가까운 빈 바닥은 L자 왼쪽 아래 모서리(트럭 옆)다. 어느 규칙이 맞는지는 그리퍼 정보 없이 정할 수 없다.

**실패 원인 분리** (`failure_attribution`)

| 단계 | 상태 |
|---|---|
| 1 객체 탐색 | OK: 후보 38개 보존 |
| 2 객체 판별 | UNVERIFIED: VLM_ACCEPTED 없음, 모두 VERIFY_REQUIRED |
| 3 의미 관계 | UNVERIFIED: "생산됨"은 명령 정보, 2D 관계만 SUPPORTED |
| 4 목적지 선택 | CONDITIONAL: 저장된 Qwen 응답(짧은 문장) + 구획 근거 |
| 5 기하학적 접근 | UNVERIFIED: 간격 13.7 / 38.0 cm, 그리퍼 도달 거리 미측정 |
| 6 경로계획 | OK (잠정 모델 안에서) |

## 7. 논문에 쓸 수 있는 것과 아직 쓸 수 없는 것

**쓸 수 있음** (구현·내부 검증 수준)
- 차체 크기를 반영한 비용 지도가 같은 지도·간선 정의에서 기존 A*와 비용이 정확히 일치하고, 도달 셀에 충돌이 없다(모델 내부 일관성).
- 관계마다 출처를 붙이는 맥락 지도 형식, 그리고 명령·VLM 진술이 기하 근거 없이 SUPPORTED가 되지 않는 규칙.
- 이동 비용이 의미 근거를 앞지르지 못하는 결정 구조와, 동률일 때 판단을 보류하는 동작(단위 테스트).
- E1 한 장의 사례: 이동 비용만 쓰는 기준선이 검증되지 않은 후보와 위치를 특정하지 못하는 영역을 고른다.

**아직 쓸 수 없음**
- 목표·목적지 선택이 정확하다거나, 의미 맥락이 선택을 개선한다는 주장. 정답이 없고, 사진이 한 장이며, 결과가 사후에 정한 on > near 규칙에 달려 있다.
- 관계 추론 정확도, 일반화 성능.
- 실제 치수 기반 주행 안전성(잠정 변환·DRAFT 지도), 픽업·배치 가능성, 메카넘 동역학.

## 8. 미검증 항목

- 사람이 확인한 큐브 라벨(8장), 실제 대상의 위치와 높이.
- 사람이 확인한 바닥 기준점과 DRAFT 구역 경계.
- 로봇 실제 외곽(바퀴 포함 여부)과 출발 방향.
- 그리퍼 장착 위치·방향·도달 거리·높이, 주행 경계 너머 조작 허용 여부.
- 빈 바닥이 실제로 비어 있는지, 콘 줄 사이로 조작이 가능한지.
- 저장된 Qwen 목적지 응답이 현재 명령의 긴 문장으로도 같은지(새 호출 필요).
- 두 번째 장면에서의 재현(현재 다른 사진에는 DINO·Qwen·바닥 변환이 없다).

## 9. 1차 프로토타입 평가

**오프라인 단일 이미지 연구 프로토타입으로서는 연결이 완성됐다.** 명령 → 저장된 VLM 해석 → 후보 → 맥락 지도 → 비용 지도 → 목표·목적지 결정 → 경로 A·B → 가상 픽업·배치가 한 번의 실행으로 재현되고, 각 단계의 기하 계산은 내부적으로 검증됐다.

**실험적으로 검증된 시스템은 아니다.** 정답이 없고, 장면이 하나이며, 좌표계는 잠정이고, 하드웨어 검증이 없다. 다음 단계로 진행하려면 사람이 확인한 큐브 라벨, 검증된 바닥 기준점, 그리퍼 치수가 필요하다.

## 10. 재현

```bash
cd /home/binci/vlm_ws/semantic_task_navigation
PY=/home/binci/miniconda3/envs/stnav_gui/bin/python; export PYTHONDONTWRITEBYTECODE=1
$PY experiments/semantic_navigation_integration/run_integration.py          # 새 results/<시각>_semnav/ 생성 (약 3초)
$PY -m pytest -q -p no:cacheprovider experiments/semantic_navigation_integration/test_semnav_offline.py
```

**입력 출처** (모두 읽기 전용)
- 이미지·지도·로봇·객체·Qwen L2·하역 구획: `task_route_integration/config.yaml` → `semantic_to_navigation_integration/results/20261118_203058_46d8`, `…/20261118_203426_260d`, `first_vlm_navigation_path/results/20261119_001504_first_path`, `semantic_evidence_verification/results/20261118_231429_bay_localization`.
- 후보: `candidate_retention_policy/results/20261119_073841_retention`.
- Qwen 보기: `qwen_crop_comparison/results/20261119_075541_crop_comparison`.
- 비교용: `task_route_integration/results/20261119_082047_route`, `placement_reach_map/results/20261119_085302_reach`.
- 구역 이름·역할: `/home/binci/vlm_ws/new_experiment_environment/my_own_ws/2026-11-15_zone_draft/semantic_zones_draft_v2.json`.

**출력** (`results/20261119_111050_semnav/`)

| 파일 | 내용 |
|---|---|
| `semantic_context_map.json` / `.png` | Exp 1 |
| `navigation_cost_map.json` / `.png` / `_arrays.npz` | Exp 2 (셀 분류, 비용, 측지 길이, 회전 가능 셀, 후보·목적지 접근) |
| `candidate_decisions.json` | 후보별 요인, 조건별 순위·결정·사유, 목적지 평가 |
| `integrated_task_plan.json` / `.png` | 통합 계획, 상태, 실패 원인 분리 |
| `ablation_results.csv` | A/B/C/D × 의미 등급 × 목표/목적지 |
| `validation.json` | 입력, Exp 1–3 검사, 코드 지문 |
| `manifest.json`, `*_snapshot.yaml`, `inputs/` | 파일 해시, 코드 해시, 설정, 입력 이미지 |

**개발 중 발견하고 고친 문제** (`superseded_runs/`)
- YAML에서 따옴표 없는 `on` 키가 불리언 `true`로 읽혔다. 설정의 키에 따옴표를 붙였다.
- HSV 전용 후보는 탐지 점수가 없다(`None`). 최저값으로 처리했다.
- 그림 함수의 시작점 키 오류를 고쳤다.
- 의미 지도에서 같은 관계가 이미지·바닥 두 기준으로 중복 생성되던 것을 하나만 남기도록 고쳤다.
- 통합 계획 그림의 범례 색이 반대로 적혀 있던 것을 고쳤다.
- 길이 0 경로를 "실패"로 보이지 않게 `ZERO_LENGTH_A`로 표시했다.

결정 규칙과 임계값은 실행 사이에 바꾸지 않았다.

**폴더 이름**: 기존 `semantic_to_navigation_integration`(v0.2 E1 실행, 여기서 재사용)과 이름이 비슷하지만 다른 폴더다. 테스트 파일 이름은 다른 실험과 겹치지 않게 `test_semnav_offline.py`로 정했다.
