# 작업 수행형 Agent 구조 설계 (2026-11-19)

- 상태: **DESIGN**. 코드 작성 없음, 모델 호출 없음, 기존 파일 수정 없음, 하드웨어 명령 없음. 이 폴더의 파일은 이 문서 하나다. 같은 날 사용자 결정에 따른 우선순위 조정은 §F에 있다.
- 표기: `CONFIRMED` 저장된 파일·코드·실행 기록으로 직접 확인 / `INFERRED` 확인한 사실에서 이어지는 해석 / `UNKNOWN` 확인하지 못함 / `탐색` 이번 조사에서 임시 스크립트로 얻은 값(저장된 실행 기록 없음, 논문 수치로 쓰지 않음)
- "AI 육안 확인"은 AI가 이미지를 보고 한 판단이며 사람의 정답이 아니다. 정답이 없는 값의 정확도는 계산하지 않았다.
- 좌표는 별도 표기가 없으면 1920×1080 왜곡 보정 사진(`raw_image_px` 계열)의 px이다. cm는 사용자가 알려 준 치수에서만 나온다.

## 0. 먼저 읽을 결론

1. **새 Agent 프레임워크는 필요 없다.** v0.2의 `AnalysisRun`(`agent/orchestrator.py`)이 이미 "고정된 도구 순서를 실행하는 Python 상태기계"이고, 모델은 도구를 고르지 않는다(`CONFIRMED`, 파일 머리말). 픽업 경로 A(시작→픽업)와 운반 경로 B(픽업→하역)를 만드는 구조(`_plan_path`)도 있다. 새로 필요한 것은 ① 작은 물체 획득 단계, ② 실제 로봇 치수와 바닥 좌표에 기반한 재검증, ③ 단계별 상태·실패 정책을 데이터로 적어 두는 것이다.
2. **가장 큰 병목은 큐브 위치 확보다.** E1에서 후보 생성은 성공했고(1순위, 3개 경로 일치) 검증 게이트가 탈락시켰다. v0.2 탐색 범위(컨베이어 ROI)는 8장 중 4장에서 큐브를 처음부터 제외한다. 같은 크기의 파란 블록(16×14 px)이 EXP110에서도 모든 조건에서 0점이었다(`CONFIRMED`).
3. **지금은 새 모델을 추가하지 않는다.** 근거는 §D. YOLO-World, YOLOE, Florence-2는 EXP110에서 이미 비교됐고, OWL-ViT 가중치는 로컬에 없다.
4. **첫 경로의 유효성은 가정 반경 35 px에서만 성립한다.** 로봇 외곽 박스(229×165 px)와 사용자 치수(35×24 cm)로 환산하면 약 6–7 px/cm이고(`INFERRED`, 아래 §B5), 35 px은 약 5–6 cm다. 실제 로봇의 반폭(12 cm)은 약 70–80 px, 보수적 원의 반경(21.2 cm)은 약 130–150 px이다. 실제 몸체 기준 재검증이 가장 먼저 필요한 작업이다.
5. **로봇 위치는 계속 추정값이다.** E1 사진의 로봇 윗면에는 마커가 없다(AI 육안 확인, `cv2.aruco` 전체 사전 검출 0건, `탐색`). 윗면의 흰 판은 작은 화면 또는 스티커로 보인다.
6. **픽업 접근에서 새로 드러난 문제.** E1 큐브 중심 (442, 424)은 DRAFT 지도의 금지 구역 Z07 안에 있고 주행 영역 Z02까지 44 px(약 6–7 cm)이다(`탐색`). 가정 도달 거리(140 px)로는 접근점이 선택되지만, 그리퍼 도달은 측정된 적이 없다(`config/planning.yaml`: `gripper_reach_px: null`). 실제 로봇은 앞쪽 그리퍼로 큐브를 향해 서야 하므로 방향이 접근 자세의 일부다.
7. **GPU는 순차 실행한다.** Qwen 4B(bf16)와 DINO 동시 추론은 12 GB GPU에서 CUDA OOM이 실측됐다(`docs/RUNBOOK.md`, `config/models.yaml`). 이 작업의 병목은 처리 시간이 아니다(§B7).

---

## A. 기존 기능 재사용표

범례: ✔ 구현·실제 사용됨 / △ 일부 또는 조건부 / ✘ 없음. 모든 경로는 `/home/binci/vlm_ws/semantic_task_navigation/` 기준.

| # | 작업 단계 | 구현 | 실제 사용 파일·함수 | 재사용 | 해결되지 않은 문제 |
|---|---|---|---|---|---|
| 1 | 자연어 → 작업 구조 | △ | `src/semantic_task_nav/agent/task_parser.py`: `rule_based_parse`, `parse_with_llm`, `build_task`(고정 `ACTION_TEMPLATES`) | 그대로 | 지원 범위가 좁다: 목적지는 `loading_area`+트럭 기준만 허용(`build_task`가 그 외를 `UNSUPPORTED_DESTINATION`으로 거부). L2("출하할 수 있는 적절한 장소")는 이 파서를 통과하지 못해, 통합 실험은 대상만 `rule_based_parse`로 해석했다 |
| 2 | 추상 명령의 목적지 추론 | ✔ | `experiments/natural_language_abstraction/nla_probe.py: run_level`, `autonomous_semantic_reasoning/probe.py: render_prompt, check_response`, 동결 프롬프트 `n0`(L1–L3) | 그대로(동결) | L2는 2건 모두 트럭 구획(L1과 IoU 0.77–0.78). L3는 2건 모두 신뢰도 high로 L1·L2와 겹치지 않는 적치 구역(IoU 0.0)을 골랐고, 탐색 실행 6건 중 보류는 0건이다(`natural_language_abstraction/REPORT_2026-11-18.md` 표 기준. 같은 보고서 본문의 "6건 중 6건"은 표와 맞지 않아 표의 수치를 썼다). 모델이 스스로 보류하지 않으므로 보류는 Python 정책이 맡아야 한다 |
| 3 | 일반 지식 조건 구분 | ✔ | `autonomous_semantic_reasoning/prompts.yaml`: 조건 A(없음)/B(5S)/C1(현장 규칙) | 그대로 | 지시문이 답을 가리키는 L1에서는 A/B/C1을 가를 수 없었다. L2·L3에서의 B/C1 비교는 하지 않았다 |
| 4 | 목적지 의미 증거 검증 | ✔ | `semantic_evidence_verification/verifier2.py: verify_case, semantic_verdict, placement_readiness`; V1 `semantic_to_navigation_integration/verify.py: decide` | V2 사용 | 저장 사례 3건 모두 계획 단계로 넘기지 못함(근거 localization 없음). 정답 영역이 없어 정확도 미측정 |
| 5 | 하역 구획 위치 확인 | △ | `semantic_evidence_verification/bay_localization.py: find_dashes, fit_two_lines, build_bay, localize` | E1 한정 | 흰 점선 L자 한 종류, 사진 한 장. 일반화 검증 없음 |
| 6 | 객체 탐색(트럭·컨베이어·콘·로봇) | ✔ | `src/semantic_task_nav/perception/grounding.py: detect_reference_objects, dino_candidates, verify_box, qwen_ground_boxes` | 그대로 | 로봇 후보는 `verify_top_k: 3` 때문에 E1에서 확인되지 않았다(점수 높은 후보 3개가 로봇팔·트럭). 크롭 확인은 같은 모델 계열 |
| 7 | **작업 대상(큐브) 탐색** | △ | `perception/target_inspection.py: make_roi, zoom_roi, hsv_candidates, merge_candidates, inspect_target` | 후보 생성(`hsv_candidates`)만 | §B4. 컨베이어 ROI 한정, VLM 단독 거부권, 모양을 묻는 질문 |
| 8 | 로봇 시작 위치 | △ | `AnalysisRun._determine_start`(박스 중심), `first_vlm_navigation_path/robot_start.py: stored_candidates, verify_in_order, start_estimates, footprint_record` | `robot_start.py` | 박스 중심은 바닥 위치가 아니다(추정). 사진에 마커 없음. 방향 없음 |
| 9 | 지도 정합 | ✔ | `navigation/map_builder.py: check_alignment, NavMap, load_map_spec`; `first_vlm_navigation_path/map_transfer.py`(P01→E1 호모그래피, 수락 기준 선언) | 그대로 | 지도 폴리곤은 사람이 확인하지 않은 DRAFT. 한 장에서만 확인. 바닥 metric 좌표 없음(`GROUND_XY`는 예약만, `contracts.py`) |
| 10 | 배치 가능 영역 | ✔ | `semantics/placement.py: placement_region` | 그대로 | 배치면과 그리퍼 도달은 측정되지 않음(`planning.yaml: placement`) |
| 11 | 접근 목표점(픽업·하역) | ✔ | `navigation/goal_selector.py: select_goal, reach_basis`(도달 = 로봇 중심에서 대상 다각형까지의 거리 ≤ `max_dist_px`) | 그대로 | 도달 거리 140/180 px는 `PROVISIONAL`(실측 아님). 방향·정차 자세 없음 |
| 12 | 경로계획·검증 | ✔ | `navigation/planner.py: plan`(A*, `SAFETY_COST_ASTAR`), `navigation/path_validator.py: validate_path`; 독립 검사 `first_vlm_navigation_path/path_checks.py` | 그대로 | 충돌 모형이 반경 35 px 원판(가정). 실제 몸체는 방향 있는 사각형 |
| 13 | 픽업 A / 운반 B 경로 구조 | ✔ | `AnalysisRun._plan_path`(`paths["A"]`, `paths["B"]`, B는 "픽업 확인 후 활성") | 그대로 | 큐브가 없으면 A가 생성되지 않는다. 실제 이미지로 A+B를 끝까지 만든 실행은 없다 |
| 14 | 인계·실행 상태 | ✔ | `robot/handoff.py: make_request, TaskExecution`(확인 출처는 사용자 시뮬레이션·외부 시뮬레이터만, 하드웨어 비활성) | 그대로 | 가상 픽업·배치를 성공으로 표시하지 않는 구조는 이미 있다 |
| 15 | Agent 상태기계 | ✔ | `agent/orchestrator.py: AnalysisRun`(`PLANS`, `REQUIRES`, `StepFailure`, 이벤트 로그). 확장 선례: `semantic_to_navigation_integration/integration.py: IntegrationRun(AnalysisRun)` | 상속으로 확장 | 의미 단계와 픽업 단계를 한 실행으로 묶은 이력이 없다 |
| 16 | 좌표 변환·크롭 출처 | ✔ | `geometry/coordinate_frames.py: RoiTransform`, 프레임 이름 `contracts.py` | 그대로 | 후보 재확인 크롭의 출처 기록 규칙이 없다(§B4) |
| 17 | 캘리브레이션·왜곡 보정 | ✔ | `geometry/calibration.py: undistort_image`; 읽기 전용 `/home/binci/camera_calibration_1920x1080/calibration_result.json` | 읽기만 | 현재 설치에서 내부 파라미터 유효성은 미검증(`docs/COORDINATE_FRAMES.md`) |
| 18 | 시각화 | ✔ | `visualization.py: export_paper_figures`, `first_vlm_navigation_path/render.py` | 그대로 | 출처별 색 규칙은 있음 |
| 19 | 마커 기반 로봇 자세 | ✘ | 참고만: `/home/binci/camera_calibration_1920x1080/aruco_pose_id0.py`(`DICT_4X4_50`, ID 0, 한 변 187 mm, 읽기 전용) | 포팅 시 출처 문서화 | 로봇에 마커가 없다. 연구 범위(워크스페이스 `AGENTS.md`)에는 포함 |

---

## B. 최종 시스템 아키텍처

### B1. 전체 데이터 흐름

굵게 표시한 단계가 새로 필요한 것이다. 나머지는 기존 코드를 호출한다.

```text
자연어 명령 ─▶ S1 작업 해석 (task_parser + 추상 명령 경로)
                 │  task_spec: 대상, 출처, 목적지(명시형/추상형), knowledge_condition
                 ▼
사진 ─▶ S2 **작은 물체 획득** (target_acquisition)  ◀─ 같은 모듈을 S9 도착 확인에도 사용
          후보 생성(HSV, 전체 바닥) → 방해물 제거 → 맥락 점수 → VLM 증거 → 상태 결정
          │  CONFIRMED | PROBABLE | AMBIGUOUS | REJECTED | NOT_VISIBLE
          ▼
        S3 목적지 의미 추론 (Qwen n0) → V2 검증 → 하역 구획 localization → 배치 영역
          ▼
        S4 **바닥 좌표(BEV)와 지도**: 호모그래피, 레이어, UNKNOWN 보존
          ▼
        S5 로봇 위치·방향·**실제 Footprint**
          ▼
        S6 접근 목표점: 픽업 접근, 배치 접근 (select_goal, 방향 포함)
          ▼
        S7 경로 A(시작→픽업), 경로 B(픽업→배치): plan + validate_path + 독립 검사
          ▼
        S8 가상 픽업·가상 배치 (handoff.TaskExecution, 성공 주장 없음)
          ▼
        S9 결과 기록: 상태, 웨이포인트, 이미지, 실패 원인
```

### B2. 모델·도구의 책임

| 도구 | 하는 일 | 하지 않는 일 |
|---|---|---|
| **Qwen3-VL-4B** | 지시문 해석, 목적지 후보 추론(근거 포함), 후보 크롭에 대한 속성 증거, 모호하면 "보류" 의견 | 좌표 계산, 경로, 충돌 판정, 도구 선택, 작은 후보의 단독 거부 |
| **Grounding DINO** | 텍스트에 대응하는 후보 박스(트럭·컨베이어·콘·로봇), 큐브는 ROI 확대 상에서 보조 후보 | 구역의 작업상 의미 결정, 17 px급 물체의 단독 확정 |
| **HSV/OpenCV** | 색·크기·모양 후보, 바닥 표시선, 호모그래피, 마스크 | 의미 판정 |
| **Python 기하** | 모든 좌표 변환, 접근점, 경로, 충돌, 검증 | 모델 출력의 무검증 사용 |
| **사람** | 지도 폴리곤 확인, 큐브 위치 정답, 로봇·그리퍼 치수 실측, 금지선 규칙 | — |

일반 지식(5S)의 위치: 목적지 추론(S3)의 선택 입력이다. 작은 물체 판정(S2)에는 넣지 않는다. 모든 Qwen 호출 기록에 `knowledge_condition: none | generic_5s | site_rules`를 남긴다(기존 조건 A/B/C1과 대응). 정보는 다섯 종류로 구분해 기록한다.

| 종류 | 예 | 기록 방법 |
|---|---|---|
| 일반 산업 지식 | "출하 = 차량이 받는 곳" | `knowledge_condition`, 모델 응답의 근거 필드 |
| 영상에서 관찰한 것 | 점선 구획, 트럭 | `contracts.OBSERVED`와 박스 |
| 사용자 지시 | "빨간 큐브", "출하" | `task_spec` |
| 현장에 명시된 안전 규칙 | "노란선 안 주행" | `MANUAL`(사람이 준 것으로만 표기) |
| AI가 추론했으나 미검증 | 모델이 주장한 근거 | `contracts.INFERRED` |

5S만으로 특정 색·구획의 기능을 확정하지 않는다. 목적지 좌표를 사람이 적어 넣지 않는다.

### B3. Agent의 작업 분할 방식

- **유지할 것: 결정론적 상태기계.** 근거는 워크스페이스 `AGENTS.md`의 역할 경계(VLM은 라벨·박스만, 도구 선택·경로·명령은 만들지 않음)와 재현성이다. 모델이 도구를 고르면 같은 입력이 다른 실행 순서를 만들 수 있다.
- **Qwen을 부르는 지점은 네 곳으로 고정한다**: 작업 해석, 목적지 추론, 작은 후보에 대한 속성 증거, 모호할 때의 의견. 어느 호출도 다음 도구를 지정하지 않는다.
- **연구 기여의 표현**: "LLM이 도구를 고른다"가 아니라 "근거 단계마다 상태를 판정하고 실패 원인에 따라 정해진 다음 행동을 하는 증거 게이트형 Agent"다. 이 선택은 사용자 결정 사항이다(§E-1).

실패 정책(상황 → Python이 하는 일). 상태 코드는 v0.2에 이미 있는 것을 우선 쓴다.

| 상황 | 상태 코드 | 다음 행동 |
|---|---|---|
| 작업 대상이 지시문에 명시됨 | — | S2에서 그 물체만 탐색 |
| 작업 대상이 여러 개 | `TARGET_AMBIGUOUS` | 하나로 확정하지 않고 후보와 근거를 모두 기록. 사용자 또는 근접 확인으로 넘김 |
| 후보 0개 | `TARGET_NOT_FOUND` (`NOT_VISIBLE`) | 종료하지 않고 다른 프레임·탐색 범위 확대를 기록 |
| Qwen과 후보 규칙이 불일치 | `PROBABLE` | 폐기하지 않고 불일치 원인을 기록, 접근만 허용 |
| 목적지 근거 부족·보류 | V2 상태(`SEMANTICALLY_SUPPORTED_PLACEMENT_UNVERIFIED` 등) | 의미 후보는 보존, 계획 중단 |
| 접근점 없음 | `ROBOT_APPROACH_NOT_FOUND` / `ROBOT_APPROACH_UNREACHABLE` | 원인(반경·도달·도달 불가)을 구분해 기록, 값을 바꿔 통과시키지 않음 |
| 경로 없음·검증 실패 | `PATH_NOT_FOUND` / `PATH_INVALID` | 원인 기록 후 중단, 충돌 조건을 풀지 않음 |

### B4. 작은 물체 획득·재확인 구조 (S2)

**현재 상태(근거)**

| 관찰 | 근거 |
|---|---|
| E1 후보 생성 성공: cand1 [434,417,451,432](17×15 px), 순위 1, 세 경로(전체 영상 DINO, ROI DINO, HSV) 일치, DINO 0.438 | `CONFIRMED`: `semantic_to_navigation_integration/results/20261118_203058_46d8/analysis/target_inspection.json` |
| 검증 게이트가 탈락시킴: Qwen 크롭 확인이 "round, not cube-shaped"로 `no` | `CONFIRMED` 같은 파일, 크롭 `analysis/roi/verify_target_1.png` |
| 게이트 구조: 81×79 px 크롭을 448 px로 확대(×5.5), 초록 사각형이 큐브 윤곽 위에 그려짐, 프롬프트가 "다른 종류의 물체거나 다른 색이면 no", `no` 한 번이 거부권 | `CONFIRMED`: `grounding.py: make_verify_crop, verify_box`, `config/models.yaml: verify_object_v1` |
| 전체 프레임 Qwen은 큐브를 "보이지 않음/위치 모름"으로 적음 | `CONFIRMED`: 저장된 L2 실행 3건, 자율 추론 평가 6건 |
| 탐색 범위: 검증된 컨베이어 주변 ROI만 사용. E1 ROI는 [0,0,758,1063] | `CONFIRMED`: `target_inspection.py: inspect_target`, 저장 기록 |
| 8장 중 4장에서 큐브가 ROI 밖(x = 1562, 1271, 1615, 1620)이고, 큐브는 8장 모두 같은 회색 격자 패드 위에 있으며 패드째 이동한다 | `탐색`(AI 육안 확인, 8장 전부 열어 봄) |
| 전체 영상 HSV 후보는 장당 15–22개이고 8장 모두 큐브를 포함. 다른 사진에서 반복되는 붉은 점을 빼면 1–4개, 패드 위 점수 순위에서 큐브가 8/8 1위 | `탐색`. 같은 8장으로 특징을 정했으므로 검증이 아니라 실현 가능성 확인이다. 패드 점수만으로는 부족(콘 줄무늬가 최대 0.24, 큐브 0.178–0.274) |
| 같은 크기의 물체가 다른 장면에서도 실패: 파란 블록 16×14 px는 DINO 84위(0.029), Qwen은 파란 드럼통 지정. test(영어)에서 block 질의 24건은 C1·C2·C6-dino 모두 0건 성공 | `CONFIRMED`: `new_experiment_environment/my_own_ws/EXP110_word_grounding/results/run01/analysis_dev_*/DEV_REPORT.md`, `run03/analysis_test_*/TEST_REPORT.md` |

해석(`INFERRED`): 17 px 물체의 모양(정육면체인지 원통인지)은 해상도로 구분되지 않는데, 게이트가 모양을 묻는다. 작은 물체에서는 VLM이 검출기도 최종 심판도 아니고, 여러 증거 중 하나로 쓰는 것이 맞다.

**새 구조**

```text
(1) 후보 생성  : v0.2 hsv_candidates 재사용을 전체 바닥(정합된 주행·작업 영역)에 적용.
                 컨베이어 ROI는 제외 조건이 아니라 점수 가중치.
(2) 방해물 제거: 이미 DINO+Qwen이 확인한 콘·로봇·트럭·컨베이어 박스와 겹치는 후보 제거.
                 여러 사진에서 같은 자리에 반복되는 점은 "순위 하향"만(정지한 큐브를 지울 수 있음).
(3) 맥락 점수  : 회색 격자 패드 위(링 점수)는 이 장면 특화 가정이므로 config에 선언하고 근거를 기록.
(4) VLM 증거   : 마커 없는 크롭 또는 윤곽 밖 모서리 표시, 질문은 "작은 붉은 블록인가, 콘 줄무늬나
                 로봇 부품 같은 다른 사물인가". 후보가 둘 이상이면 번호 붙인 한 장에서 상대 선택.
                 `no`가 구체적인 다른 사물을 지목할 때만 REJECTED. 모양만 이유면 PROBABLE.
(5) 상태 결정  : Python 규칙(아래 표). VLM 한 번의 답이 상태를 단독으로 정하지 않는다.
```

| 상태 | 의미 | 허용되는 후속 행동 | 금지 |
|---|---|---|---|
| `CONFIRMED` | 독립 증거 둘 이상(색·크기 규칙, 맥락, VLM 비거부)이 일치하고 후보가 하나 | 픽업 접근 목표·경로 A 계획 | 파지 성공 주장(파지는 근접 확인·핸드오프가 담당) |
| `PROBABLE` (`INFERRED`) | 규칙과 맥락은 통과, VLM이 모양만 불확실 | 접근 경로까지, 접근 후 근접 재확인 필요 | 파지 |
| `AMBIGUOUS` | 같은 점수대 후보 2개 이상 | 후보·근거 전부 기록, 후보 시트 상대 선택, 그래도 같으면 사용자·근접 확인 | 임의로 하나 확정 |
| `REJECTED` | 구체적인 다른 사물로 식별되었거나 알려진 방해물과 겹침 | 기록 유지(삭제 금지) | 재사용 |
| `NOT_VISIBLE` | 후보 0개 | 다른 프레임, 가림 확인, 탐색 범위 확대. 실패로 종료하지 않는다 | 후보를 만들어 내기 |

**근처 좌표 재확인**(사용자 아이디어의 구체화): 후보 주변 재확인은 항상 원본 좌표를 유지한다. `geometry/coordinate_frames.py: RoiTransform`으로 `crop_xyxy_raw`, `scale`을 기록하고, 모델이 후보 근처의 다른 박스를 내놓으면 설정한 반경(config) 안의 가장 가까운 HSV 덩어리로만 보정한다. 반경 밖이면 새 후보로 취급하지 않고 기록만 한다.

**같은 모듈을 도착 확인(S9)에도 쓴다.** 이때 탐색 범위는 하역 구획 폴리곤이고 질문은 "구획 안에 붉은 블록이 있는가"다.

### B5. 2D 지도와 경로계획

**현재**: 지도는 이미지 px의 다각형이고, 계획은 `NavMap`(셀 8 px, 반경 35 px 원판 가정)이다. `GROUND_XY`/`CALIBRATED_GROUND_MAP` 슬롯은 `contracts.py`에 예약만 있고 활성화되지 않았다(`CONFIRMED`).

**설계**

1. **바닥 호모그래피**: 검은 매트 3×2(105 cm각, 전체 315×210 cm, 사용자 제공값이며 매트 사이 간격·바닥 평면 여부는 `UNKNOWN`)의 바깥 모서리 네 점으로 이미지→바닥(cm)을 구한다. E1 사진에서 바깥 모서리는 보이지만 매트 이음선은 군데군데 희미해서 홀드아웃 기준으로 쓰기 불확실하다(AI 육안 확인). 독립 홀드아웃은 바닥에 직접 놓은 알려진 크기의 마커(기존 187 mm 설계)가 가장 확실하다 → 사용자의 물리 작업이 필요(§E-2). 수락 기준은 실행 전에 config에 선언하고, 기존 `map_transfer.py`의 방식(fit/hold-out 분리)을 따른다.
2. **BEV는 기하 전용**: 크기 검출은 원본 픽셀에서 한다. 사진을 평평하게 펴면 위쪽은 늘어나고 아래쪽은 줄어서 17 px 물체의 검출 조건이 달라진다. 펴는 것은 다각형·점·경로에만 적용한다(`INFERRED`).
3. **v0.2 계획기 재사용(수정 없음)**: `NavMap`은 `spec["image_size"]`가 입력 크기와 같기만 하면 되고 단위가 px이다(`CONFIRMED`, `map_builder.py`). 따라서 BEV px 이미지(예: 2 px/cm)를 지도 크기로 한 지도 명세를 만들면 `NavMap`·`select_goal`·`plan`·`validate_path`를 그대로 호출할 수 있다. 반경은 cm × 배율로 환산해 config에 넣는다.
4. **미확인 영역**: `NavMap`은 `traversable`에 없는 셀을 자유 공간으로 보지 않는다(`free_full = traversable & ~obstacle & ~no_go`, `CONFIRMED`). 지도에 `UNKNOWN` 레이어를 따로 기록하되 주행 가능으로 가정하지 않는다.
5. **높이 있는 물체**: 호모그래피는 바닥 평면만 정확하다. 트럭·로봇·적치물은 바닥 점유 영역(밑면)을 별도 추정해야 하고, 추정값은 `ESTIMATED`로 표기한다. 로봇 윗면 마커를 쓰면 마커 높이(약 16 cm)만큼 바닥 위치 보정이 필요하다.
6. **로봇 치수(사용자 제공)**: 35 × 24 × 16 cm. 바퀴 포함 외곽인지, 길이·폭 방향이 어느 축인지는 `UNKNOWN`이라 확인이 필요하다.
   - 보수적 원 반경 = √(35²+24²)/2 = **21.2 cm**(`CONFIRMED`, 계산).
   - 단계: ① 보수적 원판(config만 변경, 코드 수정 없음) → 통과하면 충분. ② 통과하지 못하면 방향 고정 사각형(메카넘은 방향을 유지한 채 평행 이동 가능)을 경로를 따라 쓸고 간 영역으로 검사(독립 검사 `path_checks.py`의 `swept_disc_mask`를 사각형으로 확장, 새 폴더에서). 어느 쪽도 충돌 조건을 풀거나 반경을 줄여 통과시키지 않는다.
   - 현재 첫 경로의 해석: 반경 35 px ≈ 5–6 cm이므로 실제 몸체에는 유효성을 주장할 수 없다. 스트레스 반경 82 px(외곽 박스 내접원, 반폭 12 cm에 해당)에서는 접근점이 없었다(`CONFIRMED`, `first_vlm_navigation_path/REPORT_2026-11-19.md` §8).
   - 환산 근거(`INFERRED`): 외곽 박스 229×165 px ↔ 35×24 cm에서 약 6.5–6.9 px/cm. 외곽선은 높이·바퀴 돌출을 포함해 과대이고, 배율은 바닥 위치마다 다르다. 바닥 호모그래피가 생기기 전까지는 어림값이다.
7. **픽업 접근의 방향**: 그리퍼가 앞쪽 한 곳에 있으므로 접근 자세는 (x, y, 방향)이다. 사각형 로봇의 중심은 큐브에서 최소 반길이(17.5 cm ≈ 105–120 px) + 그리퍼 도달만큼 떨어져야 한다. 현재 `select_goal`의 도달 기준(중심에서 대상까지 ≤ 140 px)은 반길이를 빼면 남는 도달이 약 20–35 px(약 3–5 cm)라서 실제 의미가 없다(`INFERRED`).
8. **경계 규칙**: E1 큐브는 흰 선과 위험표지 줄무늬로 둘러싼 생산 셀 안(Z07, DRAFT 금지 구역)에 있어서, 로봇이 경계 밖에 서서 그리퍼가 경계를 넘어 닿는 것이 허용되는지 정해져야 한다. 이는 사람이 정하는 현장 규칙이다(§E-4).

**`탐색` 값(임시 스크립트, 저장 안 됨)**: v0.2 `select_goal`을 E1 큐브 박스와 변환된 DRAFT 지도에 그대로 호출했다. 큐브 중심은 Z07 안, 주행 영역 Z02까지 44.4 px. 로봇 시작 (1699.5, 702.1) 기준으로 반경 35 px면 접근점 (540, 492), 대상에 가장 가까운 유효 셀 70.8 px. 반경 82 px면 접근점 (556, 524), 가장 가까운 유효 셀 116.4 px(가정 도달 140 px 이내). 즉 가정 도달 거리 안에서는 접근점이 생기지만, 실제 사각형 몸체의 반길이와 그리퍼 도달을 넣은 결과는 아니다.

### B6. 단계별 입출력

모든 출력은 `coordinate_frame`, `status`, `knowledge`(`OBSERVED`/`INFERRED`/`MANUAL`)를 가진다. 새 폴더에서만 쓰고 v0.2 스키마(`contracts.py`)는 수정하지 않는다.

| 단계 | 입력 | 출력(핵심 필드) |
|---|---|---|
| S1 | 지시문 | `task_spec`: `task_type`, `target{class,color,source_reference\|null}`, `destination{kind: named\|abstract, text, reference_object\|null}`, `knowledge_condition`, `ambiguity[]`, `parser{backend,raw_response}` |
| S2 | 사진, `task_spec`, 알려진 방해물 박스 | `target_acquisition`: `status`, `candidates[{id, bbox_xyxy_raw, sources[], features{color,size_px,pad_ring_score}, vlm_evidence{prompt_id,answer,raw_response,crop{crop_xyxy_raw,scale,marker}}, decision, reason}]`, `chosen_id\|null`, `next_action` |
| S3 | 사진, `task_spec` | 기존 `vlm_proposal/structured.json`, V2 판정, `bay_polygon`, `placement_region`(상태와 근거 포함) |
| S4 | 사진, 캘리브레이션, 모서리 점 | `ground_map`: `frame`(`raw_image_px`/`bev_px`), `H`, `validation{rms,holdout}`, `layers{traversable,obstacles,no_go,unknown}`, `status`(`CONFIRMED_GROUND_MAP`/`MANUAL_MAP_PRIOR`/`DRAFT_UNCONFIRMED`) |
| S5 | 사진, 지도 | `robot`: `pose{xy,heading_deg\|null,method: ESTIMATED_BBOX\|ARUCO,status}`, `footprint{shape,size_cm,status,source}` |
| S6 | 지도, 대상 위치, 로봇 | `approach_goal{kind: pickup\|place, pose{xy,heading_deg}, reach_basis{status: PROVISIONAL\|MEASURED}, stats}` |
| S7 | 지도, 시작, 접근점 | `path{label: A_to_pickup\|B_carry, status, waypoints[], validation{v02, independent}}` |
| S8 | 경로 A/B | `execution{state, mode: simulation, completed: false}`(기존 `TaskExecution`) |
| S9 | 위 전부 | `task_plan.json`, `manifest.json`(이미지 해시, 모델·코드 지문, config 스냅샷) |

웨이포인트 한 점의 형식(예):

```json
{"seq": 2, "xy": [1132.0, 652.0], "frame": "raw_image_px", "heading_deg": null,
 "segment": "A_to_pickup", "knowledge": "observed_by_model",
 "validation": {"v02": "valid", "independent": "pass", "footprint": "ASSUMED_DISC_35PX"}}
```

`heading_deg: null`은 방향이 아직 정해지지 않았다는 뜻이고, 방향이 없는 경로를 실제 주행 가능하다고 표시하지 않는다. 경로를 부드럽게 만들면 `validate_path`와 독립 검사를 다시 실행한다. `footprint` 필드가 `ASSUMED_*`인 동안 경로는 "계획 완료"이지 "주행 가능"이 아니다.

### B7. 실행 구조

- GPU 추론은 순차 실행한다. 근거는 실측이다: Qwen3-VL-4B bf16 가중치 8.3 GiB, 크롭 확인 호출 최대 8.46 GiB·2.83 s(`first_path/robot_verification.json`), 전체 장면 한 번 최대 10.07 GiB(`qwen_semantic_map_diagnostic`). Qwen+DINO 동시 추론은 OOM이었다(`RUNBOOK.md`). 모델 상주는 `gpu_residency: exclusive` 유지.
- CPU 전처리·후처리(HSV 후보, 호모그래피, 렌더링)는 병렬화할 수 있으나 이득이 작아서 **측정 전에는 하지 않는다**. 크롭 확인이 3초 안팎이라 시간이 병목이라는 근거가 없다.
- 구현 위치: 새 폴더에 `AnalysisRun`을 상속하는 얇은 러너(`IntegrationRun` 선례)를 두고 v0.2 코드는 수정하지 않는다.

---

## C. 최소 구현 순서

전체 픽업·운반·배치 **작업계획**을 이미지·데이터로 완성하는 데 필요한 것만 고른다. 각 항목은 선행 항목이 통과하기 전에는 시작하지 않는다(워크스페이스 `AGENTS.md`). 상태값은 `NOT_STARTED`/`DESIGN`.

| 순서 | 작업 | 분류 | 선행 | 통과 기준(실행 전 선언) | 상태 |
|---|---|---|---|---|---|
| 0 | **사용자 입력**(코드 아님): 로봇 외곽 치수가 바퀴 포함인지와 축 방향, 그리퍼 도달·정차 자세, 큐브 한 변(cm), E1 외 8장의 큐브 위치 확인, 금지선 침범 규칙 | **지금 필수** | — | 값이 기록됨 | NOT_STARTED |
| 1 | 큐브 후보 단계 오프라인 정식화(전체 바닥 HSV, 방해물 제거, 맥락 점수). 임시 분석을 재현 가능한 실행 기록으로 저장, 다른 세션 사진에서 일반화 확인 | **지금 필수** | 0(큐브 위치 정답) | 8/8 후보 포함, 1순위 7/8 이상. 같은 기준을 다른 세션 사진에 적용 | NOT_STARTED |
| 2 | 검증 게이트 비교(GPU 약 10분): 현재 방식, 마커 제거, 속성형 질문, 후보 시트 상대 선택. 프롬프트는 실행 전 동결, 방해물 크롭 30개 이상을 음성으로 사용 | **지금 필수** | 1 | 양성 미탐 1/8 이하이면서 음성 오탐 0. 어느 것도 못 넘으면 "이 크기에서 VLM은 게이트가 아니다"가 결론이고 근접 확인으로 넘김 | NOT_STARTED |
| 3 | 바닥 호모그래피(BEV): 모서리 4점 적합 + 독립 홀드아웃, v0.2 `NavMap` 입력으로 변환 | **지금 필수** | 0 | 수락 기준을 config에 선언(예: 홀드아웃 오차를 cm로 보고), 미달이면 `MANUAL_MAP_PRIOR` 유지 | NOT_STARTED |
| 4 | 실제 Footprint로 첫 경로 재검증(보수적 원판 → 필요 시 사각형 스윕) | **지금 필수** | 0, 3 | 통과/실패와 원인을 그대로 기록. 반경·도달 조정으로 통과시키지 않음 | NOT_STARTED |
| 5 | 픽업 접근 목표(방향 포함)와 경로 A: `select_goal`·`plan`·`validate_path` 재사용 | **지금 필수** | 2, 4 | A가 검증을 통과하거나 실패 코드로 중단 | NOT_STARTED |
| 6 | 배치 접근과 경로 B: 기존 E1 결과를 재사용하되 4의 Footprint로 재검증 | 재사용 + 재검증 | 4, 5 | 위와 같음 | NOT_STARTED |
| 7 | 러너 연결: `AnalysisRun` 상속, `task_plan.json`과 웨이포인트 스키마, 가상 픽업·배치 상태 | **지금 필수** | 5, 6 | 오프라인 테스트 통과, 가상 단계가 성공으로 표시되지 않음 | NOT_STARTED |
| — | 이미 구현되어 재사용(변경 없음): 목적지 추론 n0, V2, 하역 구획 localization, 지도 정합, `select_goal`, `plan`, `validate_path`, `TaskExecution` | 재사용 | — | — | — |
| — | 이후: 로봇 윗면 마커(ArUco)로 위치·방향 측정, L2·L3에서 일반 지식 조건(B/C1) 비교, 근접 확인(로봇 탑재 카메라, 팀원 시스템과 연동), CPU 병렬화 측정, 하역 구획 localization 일반화 | 이후 | 위 완료 | — | NOT_STARTED |
| — | **현재 제외**: 새 VLM, 검출기 학습·파인튜닝, SLAM, 새 경로계획 알고리즘, 검증기 V3, 실제 파지·주행, ROS2·CUDA 변경 | 제외 | — | — | — |

---

## D. 추가 모델 후보 평가

설치와 다운로드는 하지 않았다. 아래 "로컬 상태"는 파일 목록으로 확인한 것이다.

| 후보 | 기존 DINO 대비 풀 수 있는 문제 | 작은 물체(17 px) 적합성 | 추가 학습 데이터 | GPU·환경 | 판단 |
|---|---|---|---|---|---|
| **CLIP** | 후보 크롭과 텍스트의 유사도로 방해물과 큐브를 구분(후보 판별) | 불리하다고 본다(`INFERRED`). 위치를 찾는 모델이 아니고, 확대해도 없던 모양 정보는 생기지 않는다 | 불필요(영상–텍스트 점수) | **가중치 로컬 있음**: `/home/binci/weights/clip/ViT-B-32.pt`, `yolo_world` 환경에 `clip 1.0` | 지금 도입하지 않음. 게이트 비교(순서 2)가 모두 실패할 때 무설치로 비교 후보 추가 |
| **OWL-ViT** | CLIP 계열 특징 기반 텍스트 지정 탐지 | 이 크기에서 성능은 `UNKNOWN`. EXP110에서 비슷한 개방형 검출기가 블록에서 실패 | 불필요 | **가중치가 로컬 HF 캐시에 없음**(캐시 목록: SAM2.1-tiny, Florence-2 base/large, Grounding DINO tiny, Qwen3-VL 4B/8B). 다운로드 필요 | 이번 범위 제외 |
| **YOLO-World** | 텍스트 지정 고속 탐지 | 이미 평가됨. 합성 이미지에서 컨베이어·로봇팔 누락(`YOLO_WORLD_INTEGRATION_20260929…`). EXP110 dev(영어 196건) 질의 성공률 0.117로 DINO 0.255, Qwen 단독 0.199보다 낮고, Qwen 단독 대비 −0.082(95% CI [−0.153, −0.020])(`results/run01/scores_dev_20261114T233638/SUMMARY.md`) | 불필요 | `yolo_world` 환경에 `ultralytics 8.4.164` 있음 | 도입하지 않음 |
| **YOLOE-26s** | 같은 용도 | 이미 평가됨: dev 질의 성공률 0.122(DINO 0.255, Qwen 단독 0.199). 가중치 `/home/binci/weights/yoloe/` | 불필요 | 환경 있음 | 도입하지 않음 |
| **Florence-2** | 같은 용도 | 이미 평가됨: 0.230. 캐시에 base/large 있음 | 불필요 | 환경 있음 | 도입하지 않음 |
| **학습한 YOLO(작은 물체 전용)** | 한 가지 큐브의 반복 탐지 | 학습하면 잘할 가능성은 있으나(`INFERRED`), 이 시점에는 HSV 후보가 8/8 포함(`탐색`)이라 찾는 문제가 아니라 판별 문제다 | 필요(라벨링한 큐브 이미지) | 새 학습 환경 | 현재 제외 |

추가로 확인된 것: EXP110 dev 보고서는 "새 모델·더 큰 모델 추가는 필요 없다고 판단(INFERRED): 실패 원인이 설정·물체 크기 쪽"이라고 이미 적었다. Qwen3-VL-8B는 캐시에 있으나 bf16으로 12 GB GPU에 들어가지 않는다(같은 보고서).

결론: 이 단계에서 새 모델은 필요하지 않다. 기존 모델의 **역할 재배치**(§B4)가 먼저이고, 게이트 비교가 실패할 때만 CLIP을 무설치로 추가 비교한다.

---

## E. 한계와 사용자 결정이 필요한 것

1. **Agent 방식**: 결정론적 상태기계 유지(권장) vs 모델이 도구를 고르는 방식. 후자는 재현성과 워크스페이스 역할 경계와 충돌한다.
2. **마커**: ① 로봇 윗면에 기존 설계의 ArUco 마커 부착(위치·방향 측정, 시작점 추정 해소), ② 바닥 호모그래피 홀드아웃용 마커를 알려진 매트 위치에 배치. 둘 다 물리 작업이다.
3. **큐브 위치 확인**: 8장의 큐브 위치는 AI 육안 확인이다. 사람이 확인해야 정답으로 쓸 수 있다. 임시 몽타주: `/tmp/claude-1000/-home-binci/c8a5ca43-7978-4741-a9f7-305d8a4bf6ac/scratchpad/cube_8_photos_montage.png`(임시 폴더라 지워질 수 있음).
4. **금지선 규칙**: 생산 셀 경계(흰 선·위험표지 줄무늬)를 로봇 몸체가 넘으면 안 되는지, 그리퍼 도달은 넘어도 되는지.
5. **실측값**: 로봇 외곽 치수의 정의, 그리퍼 도달 거리, 큐브 한 변. 이 값이 없으면 `select_goal`의 도달 거리는 계속 `PROVISIONAL`이다.

**이 문서의 근거 한계**
- `탐색` 값(8장 큐브 스캔, ArUco 검출 0건, 픽업 접근 확인)은 저장된 실행 기록이 아니다. 순서 1·3·5에서 재현 가능한 실행 기록으로 만든다.
- 한 장면, 한 세션의 사진이다. 큐브 위치 라벨은 AI 육안이다.
- 모든 px→cm 환산은 한 위치(로봇 외곽 박스)에서의 어림값이다.
- 지도 폴리곤은 DRAFT, 로봇 위치는 추정이다. 따라서 어떤 경로도 "주행 가능"으로 주장하지 않고, 가상 픽업·배치도 작업 성공으로 주장하지 않는다.
- 이 문서는 `PROJECT_STATUS.md`·`NEXT_STEP.md`·`DECISIONS.md`·`EXPERIMENTS.md`를 갱신하지 않았다(이 프로젝트 폴더에는 해당 파일이 없고, 사용자 결정이 아직 없다). §E의 결정이 나오면 기록한다.

## F. 2026-11-19 우선순위 조정 (사용자 결정)

현재 목표: 실제 로봇을 구동하지 않고, 자연어 → 목적·대상·목적지 추론 → 객체 후보 탐색 → 이미지 기반 2D 지도 → 픽업·운반 접근점 → 경로 A·B → 웨이포인트와 가상 픽업·배치 시각화까지의 **이미지 기반 작업계획 프로토타입**을 완성한다. 파지·배치 성공은 주장하지 않는다. 성공 기준은 주행 성공이 아니라 **주어진 로봇 크기와 공간 제약을 반영한 2D 작업계획의 생성**이다.

| §E 항목 | 결정 | 적용 |
|---|---|---|
| 1 Agent 방식 | 새 프레임워크 보류 | 기존 결정론적 `AnalysisRun` 재사용(§B3 권장안과 같음) |
| 2 마커 | ArUco 설치 보류 | 로봇 위치는 `ESTIMATED`로 유지 |
| 3 큐브 위치 확인 | 사람 확인 정답 보류 | 큐브 위치는 AI 예비 라벨로 쓰고 정량 정확도를 주장하지 않음. §C 순서 0의 정답 확인은 선행 조건이 아님 |
| 4 금지선 규칙 | 그리퍼 경계 통과 허용 여부 보류 | 보수적 조건: **차체 점유 영역 전체가 지정된 주행 영역 밖으로 나가지 않음** |
| 5 실측값 | 그리퍼 도달·큐브 크기 실측 보류 | 필요한 값은 임의로 확정하지 않고 `ASSUMED`/`UNVERIFIED`로 기록. `select_goal`의 도달은 `PROVISIONAL` 유지 |

설계에 반영한 변경:

- **Footprint(§C 순서 4)**: 사용자 제공 35×24 cm로 초기 직사각형 Footprint를 처음부터 구성한다(보수적 원판 단계는 거치지 않음). 로봇 중심만이 아니라 차체 점유 영역이 주행 허용 경계를 넘는지 검증한다. 전후좌우·대각 이동을 고려하되 메카넘 동역학을 검증했다고 주장하지 않는다. 바퀴 포함 외곽 여부와 축 방향은 `UNVERIFIED`. v0.2의 반경 35 px은 사용하지 않는다.
- **BEV(§B5)**: 원근이 있는 사진 전체에 단일 px/cm를 적용하지 않는다. 지도 변환은 바닥 호모그래피(기존 캘리브레이션 파일은 읽기 전용)로만 한다. 이 문서의 "약 6–7 px/cm"는 로봇 위치 한 곳의 어림값이며 위험을 설명하는 데만 썼고, 계획·검증에는 쓰지 않는다.
- **지도 구분**: 주행 가능 영역 / 장애물 점유 영역 / 로봇 현재 위치 / 픽업 대상 위치 / 의미적 목적지 / 픽업·배치 접근점을 구분하고, 확인되지 않은 영역은 주행 가능으로 가정하지 않는다. 기존 수동 DRAFT 지도는 사전 정보(`MANUAL_MAP_PRIOR`)로 명시해 재사용한다.

진행 상태:

- §C 순서 1의 첫 부분을 실행했다: `experiments/cube_candidate_funnel/` (`REPORT_2026-11-19.md`). 모델 호출 없이, 기존 DINO·HSV 후보가 어느 단계에서 탈락하는지 기록했다. AI 라벨 기준으로 8장 중 4장은 컨베이어 ROI 밖이어서 ① 탐색 범위에서, E1은 Qwen 크롭 게이트(`no`)에서 탈락했고, 나머지 3장은 게이트 기록이 없어 `NOT_AVAILABLE`이다.
- 이어서: 픽업 접근 경로 A와 운반 경로 B(§C 순서 5–7). 선행 결정(E1 큐브 후보의 상태)은 §G에서 정해졌고 실행했다.
- `PROJECT_STATUS.md`·`DECISIONS.md` 등은 이 프로젝트 폴더에 없어서 이 절에 결정을 기록했다.

## G. 2026-11-19 큐브 후보 상태 결정과 작업계획 통합 (사용자 결정, 실행됨)

결정(사용자): 기존 DINO·HSV가 같은 위치를 지목한 E1 큐브 후보를 이번 작업계획의 **임시 입력**으로 쓸 수 있다. 단 `CONFIRMED`가 아니라 **`PROVISIONAL_TARGET`**이고, 저장된 Qwen 크롭 게이트의 `no`를 반드시 보존한다. 객체 탐지 정확도나 픽업 성공의 근거로 쓰지 않는다. §B4의 `PROBABLE` 대신 이 이름을 쓴다. 새 탐지 모델 설치, Qwen 프롬프트 변경, 나머지 7장의 Qwen 크롭 검증 실행은 하지 않기로 했다(보류).

실행: `experiments/task_plan_prototype/` (`REPORT_2026-11-19.md`, 실행 `results/20261119_064943_task_plan/`). 모델 호출 없이 저장된 Qwen 목적지·큐브 후보·로봇 시작점으로 경로 A, 가상 픽업, 경로 B, 가상 배치를 한 계획으로 연결했고, 보수적 직사각형 Footprint(35×24 cm + 변환 오차 3.5 cm/변)에서 A 236.3 cm, B 162.1 cm가 생성됐다. 접근 가능성은 `UNVERIFIED`(그리퍼가 메워야 할 간격 13.7 cm, 37.8 cm).

**정정**: §F와 사용자 지시는 "기존 검증된 바닥 평면 변환"을 재사용하는 것을 전제로 했다. 확인해 보니 현재 배치용 검증된 변환은 이 프로젝트에 없었다(`config/camera.yaml`: `not_available`, 2026-09-06 homography는 다른 배치·다른 undistort 프레임). 그래서 이 사진의 매트 이음선과 사용자의 105 cm로 **잠정 변환**(`PROVISIONAL_UNVALIDATED`, 맞춤 RMS 3.57 cm, 독립 이음선 편차 최대 3.5 cm, 보정 후에도 가장자리가 휜 잔여 왜곡)을 만들어 썼다. 사람이 확인한 기준점으로 검증하기 전에는 "검증된 바닥 변환"이라 부르지 않는다.

## 보존 확인

- 수정하지 않은 것: v0.2 `src/`, `config/`, V1/V2 검증기, 동결 프롬프트, 기존 실험 결과, 캘리브레이션, ROS2·CUDA·conda 환경. 읽기만 한 참조 저장소: `/home/binci/camera_calibration_1920x1080`.
- 모델은 한 번도 호출하지 않았다. 설치·다운로드 없음.
- 임시 스크립트(`aruco_probe.py`, `pickup_feasibility.py`)는 세션 임시 폴더에만 있고 프로젝트에는 없다.
