# Project Status

초기 조사일: 2026-09-16 KST

최근 환경 확인일: 2026-09-17 KST

공식 상태값: `NOT_STARTED`, `DESIGN`, `IMPLEMENTING`, `TESTING`, `PASS`, `FAIL`, `REWORK`, `FROZEN`.

## Phase Status

| Phase | Status | 확인 근거 / 미충족 gate |
|---|---|---|
| 0 Experimental Environment Freeze | DESIGN | 일부 기존 수치만 존재. 현재 camera pose, workspace, marker offset, object/gripper 조건 미확정. |
| 1 Independent Geometry Validation | DESIGN | 관련 12점 적합·LOO 코드는 존재하나 독립 20~30점 검증 구현·실험은 아직 시작하지 않음. |
| 2 Robot World Localization | IMPLEMENTING | ArUco ID/corners/camera-relative XYZ/trajectory 존재. world X/Y/yaw와 center offset·정량 검증 없음. |
| 3 External Vision Robot Pose Control | NOT_STARTED | wheel/IMU motion asset은 있으나 외부 camera pose feedback controller 없음. |
| 4 VLM Object Grounding | IMPLEMENTING | Broad scene inventory는 PARTIAL이지만 task-conditioned target grounding은 PASS. 세 partial-reference target을 각각 3회 검출했고 모든 reference IoU가 0.5 이상이었다. 다만 partial reference와 legacy geometry만 사용했으므로 최종 semantic accuracy gate는 미충족. |
| 5 Zone Grounding | IMPLEMENTING | Legacy 0.6 m grid 기반 6개 `OFFLINE_DEVELOPMENT_ZONE` 정의·검증은 PASS. 실제 physical marker, final workspace와 robot/object feasibility가 없어 final-paper zone gate는 미충족. |
| 6 Unified Workspace State | IMPLEMENTING | STEP 12 unified state, STEP 13 physical representation, STEP 14 robot-only static collision geometry는 PASS. 단, STEP 14 obstacle은 `DEVELOPMENT_SYNTHETIC`이며 gripper/object/실측 obstacle geometry가 없어 manipulation readiness는 `PARTIAL`, final planning state gate는 미충족. |
| 7 Natural-Language Task Parsing | IMPLEMENTING | STEP 11 deterministic rework는 valid integration 6/6, invalid rejection 5/5와 추가 rotate 1/1로 PASS. 다만 11-case offline development verification이며 final language benchmark는 아님. |
| 8 Gripper Hardware and Control | NOT_STARTED | 확정 hardware/config/controller와 시험 결과 없음. |
| 9 Grasp / Approach Pose | NOT_STARTED | object/gripper geometry와 시각화 구현 없음. |
| 10 Physical Feasibility Validator | NOT_STARTED | validator 구현·검증 없음. |
| 11 Path Planning | IMPLEMENTING | STEP 15 offline A*, STEP 16 static stress/sensitivity, STEP 17 invalidation, STEP 18 automatic replanning, STEP 19 logical-time persistence policy는 PASS. P1–P11에서 temporary resume/persistent replan을 검증했지만 live perception, real robot와 final physical experiment gate는 미검증. |
| 12 Grasp Integration Experiment | NOT_STARTED | 선행 Phase 미통과. |
| 13 Transport Experiment | NOT_STARTED | 선행 Phase 미통과. |
| 14 Release and Final Verification | NOT_STARTED | 성공 조건과 구현/실험 없음. |
| 15 Closed-Loop Correction | NOT_STARTED | optional; core system 이후. |
| 16 End-to-End Experiment | NOT_STARTED | 선행 Phase 미통과. |

어느 Phase도 이 Workspace에서 `PASS`하지 않았다.

## Live Third-Person Camera Scene Monitor — 2026-09-18 — SOFTWARE PASS / LIVE NOT AVAILABLE

- Fast camera loop와 future semantic/VLM loop를 분리하는 CPU OpenCV interface를 추가했다. Fixed reference, grayscale/blur/absdiff/threshold/morphology, local/global 분리, component extraction과 deterministic pixel tracking을 구현했다.
- Candidate는 `CHANGE_CANDIDATE`이며 semantic `NOT_EVALUATED`, metric geometry `NOT_AVAILABLE`, planning role `NOT_ASSIGNED`다. World coordinate, physical bbox size와 planner obstacle을 생성하지 않는다.
- F1–F8의 24 frame에서 no-change, single-frame noise, persistent/moving/disappearing rectangle, global illumination, two candidates와 merge/split을 검증했다. Timing-excluded replay는 8/8 동일했다.
- Stable local track만 semantic trigger를 만든다. GLOBAL_CHANGE는 ordinary candidate나 trigger로 만들지 않는다. Latest observation은 synchronous overwrite 방식이며 unbounded queue/thread를 만들지 않았다.
- 현재 `/dev/video*`와 `/dev/v4l/by-id/*`가 없어 intended external `Ucamera001`은 `CAMERA_NOT_FOUND`다. Built-in camera fallback과 open 시도는 없었으며 live validation은 `NOT_AVAILABLE`이다.
- 결과 JSON 12개와 annotated PNG 8개는 `/home/binci/vlm_ws/results/live_scene_monitor/`에 있다. 신규 42개를 포함해 system 및 `qwen_vl` Python에서 각각 317/317 PASS했고 STEP 12–19 산출물은 불변이다.
- Software interface는 PASS지만 live hardware 증거는 없다. STEP 20 pixel-space scene monitoring does not perform metric coordinate conversion.

## Dynamic Obstacle Persistence Policy — 2026-09-18 — PASS

- Deterministic logical time과 synthetic observation만 사용해 `BLOCKED → STOP → WAIT → RESUME/REPLAN` 정책을 추가했다. Wall clock, sleep, live camera timing은 사용하지 않았다.
- Baseline parameter는 interval 0.20 s, presence/absence confirmation 각각 2회, persistence threshold 1.00 s이며 모두 `DEVELOPMENT_PARAMETER`다.
- Raw detection과 confirmed obstacle world를 분리했다. Confirmed world가 remaining active path를 hard-block한 때만 blocking episode와 timer를 시작한다.
- P1/P4/P7은 confirmed clear 뒤 기존 revision 0 path를 재검증하고 A* 없이 resume했다. P9은 A가 clear돼도 B가 남아 resume하지 않았다.
- P2/P8/P10과 P11 두 번째 episode는 threshold 이상에서 기존 STEP 18 pipeline을 각 1회 호출해 current stopped pose `(0.61, 0.33) m`에서 revision 1 detour를 만들었다. P3은 `NO_PATH / STOP_REQUIRED`이며 partial path가 없다.
- P5 persistent off-path와 P6 soft-risk-only obstacle은 계속 이동했고 stop/replan하지 않았다. P10은 old blocker clear 후에도 revision 1을 유지했다.
- Aggregate는 73 observations, temporary/persistent episode 5/5, resume 4, planner run 5, failed replan 1, unnecessary replan 0이다. P1–P11 timing-excluded semantic replay가 모두 동일했다.
- 결과 JSON 12개와 timeline/spatial PNG 15개는 `/home/binci/vlm_ws/results/dynamic_obstacle_policy/`에 있다. 신규 46개를 포함해 system 및 `qwen_vl` Python에서 각각 275/275 PASS했고 STEP 12–18 산출물은 불변이다.
- 이 PASS는 `DEVELOPMENT TEMPORAL POLICY`만 의미한다. `STOP`은 offline navigation-policy decision이며 hardware safety stop이 아니다. Live perception, ArUco, ROS/Nav2/real robot와 manipulation은 `NOT_RUN`이다.

## Automatic Replanning After Path Invalidation — 2026-09-18 — PASS

- STEP 17 dynamic state/path invalidation과 기존 `SAFETY_COST_ASTAR`를 재사용해 offline event-driven replanning state machine을 추가했다. 새 planner나 smoothing은 만들지 않았다.
- Current robot pose는 event의 active-path waypoint index에서 가져온다. R2/R3/R6/R7 첫 event는 revision 0의 index 20, `(0.61, 0.33) m`에서 original goal `(0.95, 1.50) m`까지 재계획했다. Original mission start를 재사용하지 않았다.
- R1 away obstacle과 R5 soft-cost-only event는 path가 `VALID`여서 planner를 실행하지 않고 revision 0을 유지했다. R5 remaining-path clearance/risk는 0.105→0.035 m, 0.443781→0.662470이었다.
- R2 ADD와 R3 MOVE는 future path를 차단해 revision 1로 전환했다. 새 path는 2.092548 m, clearance 0.075 m, risk 0.707693이며 독립 검증을 통과했다.
- R4 full-width barrier는 `NO_PATH`를 반환해 `REPLAN_FAILED / STOP_REQUIRED`가 됐고 partial path나 새 active path를 만들지 않았다.
- R6은 두 replan에서 lineage `0→1→2`를 보존했다. Revision 2 path는 1.637990 m, clearance 0.105 m, risk 0.446781이었다.
- R7은 blocker 제거 후 revision 1 remaining path가 계속 valid이므로 planner를 실행하지 않았고 revision 0으로 자동 복귀하지 않았다.
- Robot pose/goal collision은 각각 `ROBOT_POSE_INVALID`와 `GOAL_BLOCKED`로 명시적으로 stop하며 nearest-free snapping을 하지 않는다.
- 결과 JSON 8개와 revision별 visualization 16개는 `/home/binci/vlm_ws/results/automatic_replanning/`에 있다. 기존 189개와 신규 40개 test가 system 및 `qwen_vl` Python에서 각각 229/229 PASS했고 STEP 12–17 산출물은 불변이다.
- 이 PASS는 offline automatic replanning baseline만 의미한다. Obstacle persistence/timeout/debounce, live camera/VLM, ArUco localization, ROS/Nav2/real robot와 manipulation은 `NOT_RUN`이다.

## Dynamic Obstacle State Update + Current Path Invalidation — 2026-09-18 — PASS

- `DEVELOPMENT_SYNTHETIC` offline D1–D6 scenario에서 stable obstacle ID, `ADD`/`REMOVE`/`MOVE`와 deterministic `world_revision` 0→1→2를 구현했다.
- 각 scenario는 기존 STEP 15 `SAFETY_COST_ASTAR`로 초기 path를 한 번 생성했다. Event 이후 동일한 `path_grid`/`path_world`만 재검증했고 post-event A* 실행은 0회다.
- 각 event마다 raw metric obstacle state에서 기존 collision world를 재구성하고 90×60 binary occupancy와 safety layer 전체를 rebuild했다. Robot footprint expansion은 collision builder에서 한 번만 적용했고 additional inflation은 0이다.
- Path validator는 valid robot-center workspace, waypoint collision, 8-connected adjacency, 연속 metric edge–expanded obstacle intersection, diagonal corner-cut을 검사하고 최초 invalid waypoint/edge 및 obstacle provenance를 반환한다.
- D1은 `VALID`, D2/D3은 `BLOCKED`, D4/D5는 `BLOCKED→VALID`, D6은 `VALID`였다. 모든 blocked event는 `replan_required=true`, valid event는 false다.
- D2/D3/D4/D5의 차단 revision은 waypoint 50/edge 49와 각 dynamic obstacle ID를 기록했다. D6은 clearance 0.065→0.035 m, risk 0.487105→0.705794로 변했지만 hard collision이 없어 `VALID`를 유지했다.
- Timing은 이번 실행에서 collision rebuild 0.234–0.312 ms, grid rebuild 7.379–8.517 ms, validation 0.090–0.280 ms였으며 benchmark가 아닌 volatile observation이다.
- 결과 JSON 7개와 revision별 visualization 14개는 `/home/binci/vlm_ws/results/dynamic_path_invalidation/`에 있다. 기존 153개와 신규 36개 test가 system 및 `qwen_vl` Python에서 각각 189/189 PASS했고 STEP 12–16 산출물은 불변이다.
- 이 PASS는 offline dynamic state update/current-path invalidation만 의미한다. Automatic replanning, live camera/VLM event source, ArUco, real robot와 manipulation은 `NOT_RUN`이다.

## Legacy Offline Geometry Software Verification — 2026-09-17 — PASS

- `photo_04.png`를 저장된 `new_camera_matrix`로 undistort한 결과는 `photo_04_undistorted.png`와 pixel-exact하게 일치했다: mean absolute difference 0, max difference 0, changed pixel ratio 0.
- 저장된 raw 12점의 undistorted pixel 및 world 변환을 재현했다: undistortion reproduction max 0 px, world reproduction max 0 m.
- 같은 12점에 대한 legacy fitting error 재계산값은 mean 5.34 mm, RMSE 5.74 mm, max 8.80 mm다.
- 동일 명령을 재실행해 image, JSON 2개, CSV의 SHA-256이 각각 동일함을 확인했다.
- 구현은 `/home/binci/vlm_ws/src/vlm_reposition/geometry/legacy.py`, 검증 명령은 `/home/binci/vlm_ws/scripts/verify_legacy_geometry.py`, 결과는 `/home/binci/vlm_ws/results/geometry_legacy_regression/`에 있다.
- 이 PASS는 `LEGACY_OFFLINE_GEOMETRY`의 software reproduction을 뜻한다. 독립 reference point를 사용한 정확도 검증이 아니므로 Phase 1 상태는 `DESIGN`으로 유지한다.

## Qwen Object Grounding Software Integration — 2026-09-17 — PASS

- 기존 `qwen_vl` 환경과 local-only `Qwen/Qwen3-VL-4B-Instruct` cache를 재사용했다. 새 package나 model을 설치·다운로드하지 않았다.
- 저장된 red-cup bbox `[447,527,483,595]`를 새 pipeline에 입력해 raw bottom-center `(937.44,674.135) px`, undistorted `(902.753845,744.592834) px`, legacy world `(0.267293,0.657320) m`를 계산했다.
- fresh inference 3회는 모두 bbox `[448,525,483,595]`, bottom-center `(938.448,674.135) px`, world `(0.269240,0.657628) m`로 동일했다.
- inference latency는 2.717 s, 2.051 s, 2.054 s였고 model load는 3.514 s였다.
- Qwen 원문, 각 좌표 단계, latency, summary와 visualization을 `/home/binci/vlm_ws/results/qwen_grounding/`에 저장했다.
- `qwen_vl`의 OpenCV 5.0.0에서도 STEP 04 image/12-point geometry regression이 PASS했다.
- 이 PASS는 single-target semantic→pixel→metric software integration을 뜻한다. 독립 object position ground-truth가 없으므로 Phase 4는 `IMPLEMENTING`으로 유지한다.

## Multi-Entity Semantic Grounding Integration — 2026-09-17 — PARTIAL

- `photo_04.png`에 같은 structured scene prompt를 3회 실행했고 매번 11개 entity를 동일한 순서·label·classification·bbox로 반환했다.
- Python이 각 run에서 `entity_001`~`entity_011`을 결정론적으로 할당했다. label+IoU 0.5 matching에서 11개가 모두 common entity였고 bbox IoU와 world X/Y run-to-run variation은 각각 1.0과 0이었다.
- 각 run의 geometry status는 `VALID` 3, `OUTSIDE_LEGACY_WORKSPACE` 8, `INVALID` 0이었다. 범위 밖 좌표는 clamp하지 않았다.
- legacy red cup, white bag, trophy-like object bbox와 fresh bbox의 최고 spatial IoU는 각각 0.9452, 0.9100, 0.8730이었다.
- 하지만 fresh model은 세 물체의 실제 이름 대신 모두 generic `workspace_element` label을 사용했다. 이는 semantic grounding 품질의 명확한 제한이다.
- obstacle bbox는 physical polygon/footprint로 변환하지 않았고 occupancy map도 생성하지 않았다.
- 결과는 `/home/binci/vlm_ws/results/multi_entity_grounding/`에 있으며 software integration 기록일 뿐 semantic/detection/final-paper accuracy가 아니다.

## Semantic Label Specificity and Active Workspace — 2026-09-17 — PARTIAL

- legacy `photo_04_objects_world.json`에서 bbox를 읽어 red cup, white bag, gold trophy-like object 세 개만 포함하는 `PARTIAL_SEMANTIC_REFERENCE`를 만들었다. 나머지 entity는 `UNREVIEWED`다.
- VLM output에서 `navigation_role`을 제거하고 specific `label`, `semantic_type`, `bbox_2d`만 수용한다. Target/obstacle/path role은 perception 책임이 아니다.
- 수정 prompt는 현재 reference 정답 이름을 노출하지 않았으며 3회 모두 11개 entity, `SPECIFIC` 11, `GENERIC` 0, `UNKNOWN` 0을 동일하게 반환했다.
- red cup reference 위치는 `red mug`로 IoU 0.9444에 association됐으나 등록 alias와 달라 `LABEL_MISMATCH`, white bag은 최고 IoU 0.0652로 `NO_BBOX_MATCH`, trophy는 `trophy`/IoU 0.9153으로 PASS했다.
- reference result는 매 run 1/3 PASS였으므로 STEP 결과는 PARTIAL이다. Prompt가 generic label 문제는 해결했지만 세 known object semantic verification gate는 통과하지 못했다.
- active workspace status는 매 run `INSIDE` 3, `OUTSIDE` 8, `INVALID_GEOMETRY` 0이었다. OUTSIDE entity는 scene에서 삭제하지 않았다.
- 결과는 `/home/binci/vlm_ws/results/semantic_refinement/`에 있으며 full scene semantic accuracy나 final-paper metric이 아니다.

## Task-Conditioned Target Grounding — 2026-09-17 — PASS

- Scene Inventory와 Task-Conditioned Target Grounding을 분리했다. Broad scene 결과가 모든 가능한 물체를 열거해야 한다고 가정하지 않으며, task mode에는 reference 좌표가 아닌 canonical target phrase만 제공했다.
- partial reference의 `red cup`, `white bag`, `gold trophy-like object`를 각각 3회 실행했고 9/9 run이 단일 `TARGET_FOUND`와 `REFERENCE_POSITION_MATCH`를 반환했다.
- reference IoU는 red cup 0.9452, white bag 0.9341, trophy-like object 0.9234였고 각 target의 run-to-run bbox IoU는 모두 1.0, world X/Y peak-to-peak은 모두 0 m였다.
- Broad scene에서 `NO_BBOX_MATCH`였던 white bag을 task-conditioned mode에서 3/3회 복구했다.
- 단일 후보가 아니면 좌표를 선택하지 않는다: 0개는 `TARGET_NOT_FOUND`, 2개 이상은 `TARGET_AMBIGUOUS`다.
- label/alias 판정과 reference position association을 별도 status로 기록한다. `red mug`는 자동 alias가 아니라 사람 검토가 필요한 후보로만 취급하며, 이번 task-conditioned 출력은 `red cup`이어서 alias 후보 추가는 없었다.
- STEP 04 geometry와 STEP 05 loader/변환을 그대로 재사용했고 bbox 좌표를 clamp하지 않았다. White bag world point `(1.387814, 0.584694) m`는 configured X 범위 밖이므로 `OUTSIDE`로 보존했다.
- 결과는 `/home/binci/vlm_ws/results/task_conditioned_grounding/`에 있다. 이 PASS는 partial-reference task-conditioned software verification이며 full scene semantic accuracy나 final-paper metric이 아니다.

## Offline Metric Goal Zone Definition — 2026-09-17 — PASS

- `LEGACY_WORKSPACE_COORDINATE` X 0.0–1.2 m, Y 0.0–1.8 m의 기존 0.6 m grid를 그대로 사용해 `zone_A`~`zone_F` 여섯 cell을 `OFFLINE_DEVELOPMENT_ZONE_SET`으로 정의했다.
- 모든 ID/alias가 unique하고 polygon은 finite·convex·workspace 내부이며 center는 자기 polygon 내부다. Polygon interior overlap은 0 m²이고 공유 edge만 허용한다.
- Alias는 deterministic exact lookup만 수행한다. `A`, `A zone`, `zone A`, `A 구역`, `A구역`은 `zone_A`로 resolve되지만 방향/지시 표현은 `ZONE_NOT_RESOLVED`다.
- 공유 boundary 위의 point는 한 zone을 선택하지 않고 `BOUNDARY`와 접한 모든 zone ID를 반환한다.
- STEP 08 run01 world coordinate를 read-only로 조회한 결과 red cup은 `zone_C`, gold trophy-like object는 `zone_F`, white bag은 `OUTSIDE_WORKSPACE`였다. White bag을 nearest zone으로 할당하지 않았다.
- 기존 raw→world geometry는 수정하지 않았다. Inverse Homography는 undistorted image와 metric diagram visualization에만 사용했다.
- 결과는 `/home/binci/vlm_ws/results/goal_zones/`에 있다. 이 PASS는 virtual offline zone model 검증이며 physical/final-paper zone 검증이 아니다.

## Natural-Language Task Parser + Resolution — 2026-09-17 — PARTIAL

- Existing local-only `Qwen/Qwen3-VL-4B-Instruct`를 text-only language parser로 재사용했다. 새 model/package download는 없었다.
- LLM output은 `action`, `target_phrase`, `destination_phrase` 세 field만 허용하며 Python이 schema/action/missing/ambiguity를 검증하고 STEP 09 resolver로 destination을 결정한다.
- 11-case development set에서 action field 6/6, destination zone 6/6, English valid 3/3, invalid rejection 4/5가 맞았다.
- 한국어 valid 3개는 action과 zone을 맞혔지만 target을 `빨간 컵`, `흰색 가방`, `금색 트로피`로 유지해 STEP 08의 canonical English grounding record와 연결되지 않았다. 따라서 전체 valid integration은 3/6이다.
- `빨간 컵을 회전시켜`는 `reposition`으로 과도하게 분류되어 기대 `UNSUPPORTED_ACTION` 대신 `TASK_INCOMPLETE/MISSING_DESTINATION`이 됐다.
- 대표 3문장 각 3회에서 action/target/destination/zone 출력은 run-to-run 동일했다. Repeatability는 관찰됐지만 correctness를 의미하지 않는다.
- English red-cup integration은 STEP 08 world `(0.269240, 0.657628) m`, `INSIDE`와 STEP 09 `zone_A` polygon/center를 결합했다. Feasibility는 `NOT_EVALUATED`로 보존했다.
- 결과는 `/home/binci/vlm_ws/results/task_parser/`에 있다. Prompt development의 앞선 PARTIAL raw 결과도 `task_parser_attempt01/`, `task_parser_attempt02/`에 보존했다.

## Deterministic Task Canonicalization + Action Guard — 2026-09-17 — PASS

- STEP 10의 기존 파일과 raw 결과를 보존한 채 중단 지점부터 rework했다. `task_language.json`, parser config 연결과 추가 English rotate case는 선행 변경을 그대로 사용했다.
- Qwen text-only output을 canonical value가 아닌 `action_phrase`, `target_phrase`, `destination_phrase` surface extraction으로 제한하고, Python이 action/target/zone을 결정한다.
- Human-reviewed development mapping으로 `빨간 컵`→`red cup`, `흰색 가방`→`white bag`, `금색 트로피`→`gold trophy-like object`를 resolve했다. Unknown explicit target은 `OPEN_VOCABULARY`, referential target은 `TARGET_AMBIGUOUS`다.
- Raw instruction unsupported-action guard가 model output보다 먼저 적용된다. Korean `회전시켜`와 English `Rotate`는 모두 `UNSUPPORTED_ACTION`이고 grounding/zone/path는 `NOT_RUN`이다.
- 기존 core metrics는 valid integration 6/6, action 6/6, target canonical phrase 6/6, destination zone 6/6, invalid rejection 5/5였다. 추가 English rotate regression도 1/1 PASS했다.
- Korean 세 valid case가 모두 STEP 08 `TARGET_FOUND`에 연결됐다. Red cup `(0.269240, 0.657628) m`/`INSIDE`, white bag `(1.387814, 0.584694) m`/`OUTSIDE`, trophy `(0.699674, 1.467030) m`/`INSIDE`를 변경 없이 보존했다.
- 대표 3문장 각 3회에서 action surface/canonical action/target surface/grounding phrase/destination surface/zone이 모두 동일했다.
- 결과는 `/home/binci/vlm_ws/results/task_parser_rework/`에 있다. 첫 STEP 11 PARTIAL attempt도 `/home/binci/vlm_ws/results/task_parser_rework_attempt01/`에 보존했다.

## Unified Workspace / Task State Construction — 2026-09-17 — PASS

- 새 inference, geometry estimation 또는 task parsing 없이 STEP 11 validated task, STEP 08 target grounding, STEP 09 zone config와 STEP 07 `semantic_run01.json`의 11개 entity를 read-only input으로 결합했다.
- `red cup→zone_A`, `white bag→zone_B`, `gold trophy-like object→zone_F` 세 state가 모두 `VALID_REPRESENTATION`이다. White bag의 `OUTSIDE` 상태도 failure로 바꾸지 않고 보존했다.
- 모든 metric source는 `LEGACY_WORKSPACE_COORDINATE`/`m`로 검증했고 workspace bounds X 0.0–1.2 m, Y 0.0–1.8 m를 `zones.json`에서 읽었다.
- Target-scene bbox association은 red cup→`entity_007`/IoU 0.9437, trophy→`entity_009`/IoU 0.9717이다. White bag은 최고 IoU 0.0632로 연결하지 않았으며 target grounding 자체는 authoritative하게 유지했다.
- Invalid task, target-not-found, unknown destination, frame mismatch와 non-finite coordinate를 명시적으로 거부한다. Feasibility는 `NOT_EVALUATED`, planning은 `NOT_RUN`, robot은 `NOT_AVAILABLE`이다.
- Metric visualization과 state/summary JSON은 `/home/binci/vlm_ws/results/workspace_state/`에 있다. 기존 50개와 새 10개 test가 system 및 `qwen_vl` 환경에서 60/60 PASS했다.
- 이 PASS는 offline representation software integration이며 STEP 07 scene의 PARTIAL semantic quality, legacy geometry와 미검증 physical feasibility를 개선하지 않는다.

## Physical Geometry + Planning-Relevant Footprint Layer — 2026-09-17 — PASS

- `/home/binci/vlm_ws`, `/home/binci/qwen_vision_test`, `/home/binci/mecanum_ros2`를 read-only audit했다. Team repository는 commit `3c710ab27209eaa196facb6a88507c42447837a8` 상태의 reference로만 사용하고 수정하지 않았다.
- 기존 Nav2 `base_link` footprint `[[0.145,0.123],[0.145,-0.123],[-0.145,-0.123],[-0.145,0.123]] m`를 재사용했다. Rectangle extent는 0.290×0.246 m이며 robot geometry readiness는 `READY`다.
- Nav2 `footprint_padding: 0.01 m`와 `inflation_radius: 0.55 m`는 물리 치수가 아니므로 config에 승격하거나 적용하지 않았다.
- Gripper front extension/open/closed width/holding offset, 세 target object footprint, selected chair/desk obstacle polygon은 확인된 source가 없어 `UNKNOWN`/`NOT_READY`로 보존했다. Qwen bbox에서 physical size를 계산하지 않았다.
- Planning role은 semantic type과 분리했다. Red/trophy associated entity는 `TASK_TARGET`, person은 `DYNAMIC_ENTITY`, furniture는 `STATIC_OBSTACLE_CANDIDATE`, 나머지는 `UNASSIGNED`다. 어떤 scene point에도 임의 footprint를 만들지 않았다.
- 세 STEP 12 state를 overwrite하지 않고 별도 physical state로 통합했다. Representation은 `VALID`, overall planning geometry readiness는 `PARTIAL`, collision checking과 planning은 `NOT_RUN`이다.
- 결과는 `/home/binci/vlm_ws/results/physical_geometry/`에 있다. 기존 60개와 새 12개 test가 system 및 `qwen_vl` 환경에서 72/72 PASS했다.

## Static Obstacle Representation + Robot Collision Geometry — 2026-09-17 — PASS

- 기존 Nav2 `base_link` robot footprint 0.290×0.246 m를 yaw 0의 고정 직사각형으로 재사용했다. Half extent는 X 0.145 m, Y 0.123 m다.
- 기존 legacy workspace X 0.0–1.2 m, Y 0.0–1.8 m에서 robot-center valid bounds는 X 0.145–1.055 m, Y 0.123–1.677 m다.
- 실측 obstacle polygon이 없어 명시적 `DEVELOPMENT_SYNTHETIC` rectangle X 0.45–0.65 m, Y 0.65–1.15 m 하나를 별도 config에 정의했다. 이는 VLM detection이나 실제 환경 측정값이 아니다.
- Robot footprint로 configuration-space 확장한 collision obstacle은 X 0.305–0.795 m, Y 0.527–1.273 m다. 추가 safety margin, padding, inflation은 0이다.
- Workspace/obstacle 경계 접촉은 `COLLISION`이다. 연속 metric point query로 내부·외부·접촉과 frame/unit/non-finite/invalid geometry를 검증했다.
- Robot-only static planning geometry는 `READY`지만 overall manipulation geometry는 STEP 13의 `PARTIAL`을 유지한다. Occupancy grid, A*, path, dynamic replanning과 manipulation은 구현·실행하지 않았다.
- 결과는 `/home/binci/vlm_ws/results/collision_geometry/`에 있다. 기존 72개와 새 19개 test가 system 및 `qwen_vl` Python에서 각각 91/91 PASS했다.

## Binary Occupancy Grid + Safety Cost Layer + 8-Connected A* Baseline — 2026-09-17 — PASS

- 저장된 STEP 14 collision world를 source of truth로 사용해 0.02 m cell-center grid 90×60을 생성했다. 5,400개 cell 중 FREE 2,638, OCCUPIED 2,762이며 additional inflation은 0이다.
- Hard occupancy와 soft normalized safety risk를 분리했다. Risk는 robot-expanded collision obstacle 및 valid robot-center workspace boundary까지의 clearance로 계산하며 radius 0.20 m, weight 2.0은 `DEVELOPMENT_PARAMETER`다.
- Start `(0.20,0.30) m`, goal `(0.95,1.50) m`은 독립 `DEVELOPMENT_DEFINED` 값이다. Half-open containing-cell policy로 각각 center `(0.21,0.31) m`, `(0.95,1.51) m`에 mapping됐다.
- 고정 neighbor order `N, NE, E, SE, S, SW, W, NW`와 diagonal corner-cut prevention을 사용했다.
- Distance-only path는 1.740833 m, minimum clearance 0.015 m, accumulated risk 1.355386, 81 waypoints, 1,286 nodes였다.
- Safety-cost path는 1.846274 m, minimum clearance 0.065 m, accumulated risk 0.487105, 90 waypoints, 1,765 nodes였다. 두 path 모두 collision-free다.
- Planning time은 해당 실행에서 distance-only 4.401 ms, safety-cost 5.868 ms였으며 volatile metric으로 결정론 비교에서 제외했다.
- Occupancy/cost/path semantic reconstruction은 동일했고 기존 STEP 12/13/14 SHA-256도 유지됐다. 기존 91개와 신규 31개 test가 system 및 `qwen_vl` Python에서 각각 122/122 PASS했다.
- 이는 offline development baseline이며 A*를 연구 novelty로 주장하지 않는다. Path smoothing, dynamic replanning, ArUco, real robot와 manipulation은 `NOT_RUN`이다.

## Static Path Stress Test + Parameter Sensitivity — 2026-09-18 — PASS

- STEP 15 planner 코드를 변경하거나 복제하지 않고 기존 collision world builder, grid, safety layer와 두 A*를 scenario runner에서 재사용했다.
- `DEVELOPMENT_SYNTHETIC` scenario 6개를 실행했다: S1 baseline, S2 shifted obstacle, S3 overlapping multi-obstacle, S4 footprint expansion 후 0.030 m narrow passage, S5 full-width blocked, S6 alternate endpoints.
- PATH_EXISTS 5개에서 두 planner의 10/10 결과가 독립 waypoint/continuous collision/workspace/adjacency/corner-cut validation을 통과했다. S5는 두 planner 모두 partial path 없이 `NO_PATH`였다.
- S1 collision world와 두 semantic path가 저장된 STEP 15 결과와 동일했다. 각 scenario를 두 번 생성한 occupancy/risk/path semantic 결과도 동일했다.
- Resolution 0.01/0.02/0.04 m, safety radius 0.10/0.20/0.30 m, safety weight 0/0.5/1/2/4를 one-factor-at-a-time으로 평가했다. 총 11 parameter settings와 14 planner runs이다.
- Safety radius와 weight는 hard occupancy를 바꾸지 않았고 모든 grid의 additional inflation은 0이었다. Weight 0의 safety objective는 distance-only objective와 같았다.
- 관측된 baseline weight sweep에서 weight 0→4는 accumulated risk 1.355386→0.474359, path length 1.740833→1.874558 m였다. 이는 tested development scenario의 observation이며 일반 보장이나 parameter optimization이 아니다.
- Safety-radius sweep의 minimum clearance는 모두 0.065 m였다. Fixed start endpoint의 boundary clearance가 minimum을 제한했으므로 radius 증가가 항상 clearance를 증가시킨다는 결론을 내리지 않는다.
- 결과는 `/home/binci/vlm_ws/results/path_planning_stress/`에 있다. 기존 122개와 신규 31개 test가 system 및 `qwen_vl` Python에서 각각 153/153 PASS했다.
- Dynamic replanning, live perception, ArUco, real robot, smoothing, rotation-aware planning과 manipulation은 `NOT_RUN`이다.

## Camera Setup Verification — 2026-09-17

### CONFIRMED

- 현재 V4L/USB에서 확인된 camera는 내장 `HP True Vision FHD Camera` (`30c9:00d9`)뿐이다.
- `/dev/video0`은 RGB capture node이며 MJPG 1920×1080 @ 30 fps를 지원한다.
- `/dev/video1`은 `/dev/video0`의 metadata node다.
- `/dev/video2`는 GREY 400×400 @ 15 fps capture node이고 `/dev/video3`은 그 metadata node다.
- 내장 RGB camera의 stable path는 `/dev/v4l/by-id/usb-DUKUF0A8PLMYR8_HP_True_Vision_FHD_Camera_200901010001-video-index0`이다.
- 기존 코드가 가리키는 외장 camera path `usb-HD_Web_Camera_HD_Web_Camera_Ucamera001-video-index0`은 현재 검출되지 않았다.
- default environment는 `/usr/bin/python3` 3.10.12, OpenCV 4.5.4다.
- 연구용 외장 camera가 식별되지 않아 capture test는 수행하지 않았고 probe image도 생성하지 않았다.

### INFERRED

- `camera_calibration_ros2.yaml`의 `camera_name: hd_web_camera_ucamera001`과 기존 Qwen 코드의 by-id path를 근거로, 기존 calibration은 현재 내장 HP camera가 아니라 별도 `HD Web Camera ... Ucamera001`을 대상으로 했을 가능성이 높다.

### UNKNOWN

- 연구용 외장 camera의 현재 연결 상태, 실제 stable device path, capability와 frame 안정성.
- 기존 calibration과 앞으로 연결할 외장 camera가 동일한 물리 장치인지 여부.
- 최종 camera 높이/각도, workspace 원점·축·경계와 mount 재현성.

### HUMAN_ACTION_REQUIRED

1. 연구에 사용한 외장 USB webcam을 laptop에 연결한다.
2. 다음 조사 전에는 camera를 최종 위치로 확정하거나 기존 homography가 유효하다고 가정하지 않는다.

### STEP 02 Recheck — PARTIAL

- 2026-09-17 재탐색에서도 `v4l2-ctl`과 `lsusb`에는 내장 HP camera만 확인됐다.
- 외장 camera identity/capability를 확인할 수 없어 OpenCV capture를 실행하지 않았다.
- `datasets/raw/camera_setup/`과 `scripts/capture_still_image.py`는 생성하지 않았다.
- 다음 blocker는 연결했다고 가정한 외장 camera가 OS USB/V4L 목록에 나타나지 않는 문제다.

## Offline Reference Image Audit — 2026-09-17 — PASS

### Inventory Summary

- 총 image: 15
- `RAW_INPUT`: 8
- `UNDISTORTED_INPUT`: 1
- `QWEN_OUTPUT`: 2
- `HOMOGRAPHY_VISUALIZATION`: 3
- `UNKNOWN`: 1

| File | Resolution / channels | Bytes | Modified (KST) | Likely role | Evidence |
|---|---:|---:|---|---|---|
| `photo_01.png` | 2016×1133×4 | 3,334,217 | 2026-09-04 02:03:40 | RAW_INPUT | 비주석 workspace image; code reference 없음 |
| `photo_02.png` | 2016×1133×4 | 3,360,167 | 2026-09-04 02:03:42 | RAW_INPUT | 비주석 workspace image; code reference 없음 |
| `photo_03.png` | 2016×1133×4 | 3,355,142 | 2026-09-04 02:03:42 | RAW_INPUT | 비주석 workspace image; code reference 없음 |
| `photo_04.png` | 2016×1133×4 | 3,357,172 | 2026-09-04 02:03:42 | RAW_INPUT | Qwen, point collection, undistortion, grid, visualization 공통 입력 |
| `photo_05.png` | 2016×1133×4 | 3,503,748 | 2026-09-04 02:03:44 | RAW_INPUT | 비주석 workspace image; code reference 없음 |
| `photo_06.png` | 2016×1133×4 | 3,418,636 | 2026-09-04 02:03:44 | RAW_INPUT | 비주석 workspace image; code reference 없음 |
| `photo_07.png` | 2016×1133×4 | 3,426,100 | 2026-09-04 02:03:46 | RAW_INPUT | 비주석 workspace image; code reference 없음 |
| `photo_08.png` | 2016×1133×4 | 3,427,924 | 2026-09-04 02:03:46 | RAW_INPUT | 비주석 workspace image; code reference 없음 |
| `photo_04_undistorted.png` | 2016×1133×3 | 3,137,750 | 2026-09-06 17:56:11 | UNDISTORTED_INPUT | `undistort_photo04.py` 출력이며 Homography JSON의 직접 입력 좌표계 |
| `photo_04_bbox_world.png` | 2016×1133×3 | 4,281,449 | 2026-09-06 15:34:14 | QWEN_OUTPUT | bbox/world overlay 출력 |
| `photo_04_multi_bbox_world.png` | 2016×1133×3 | 4,279,505 | 2026-09-06 11:53:43 | QWEN_OUTPUT | multi-object bbox/world overlay 출력 |
| `photo_04_grid_lines.png` | 2016×1133×3 | 4,162,971 | 2026-09-06 12:30:39 | HOMOGRAPHY_VISUALIZATION | floor/grid overlay 출력 |
| `photo_04_homography12_visualized.png` | 2016×1133×3 | 4,266,700 | 2026-09-06 15:39:46 | HOMOGRAPHY_VISUALIZATION | 12-point Homography overlay 출력 |
| `photo_04_marked.png` | 2016×1133×3 | 4,268,749 | 2026-09-06 04:49:12 | HOMOGRAPHY_VISUALIZATION | 좌표 quadrilateral이 보이나 생성 script는 확인되지 않음 |
| `qwen_frame.jpg` | 640×480×3 | 128,604 | 2026-08-30 20:41:09 | UNKNOWN | 비주석 frame으로 보이나 code/provenance reference 없음 |

모든 image는 OpenCV로 읽혔고 dtype은 `uint8`이었다. `photo_01.png`~`photo_08.png`의 4 channels는 `IMREAD_UNCHANGED` 기준이며 기존 script의 기본 `cv2.imread()`에서는 BGR 3 channels로 읽힌다.
여기서 `RAW_INPUT`은 비주석 원본 입력이라는 뜻이며 sensor RAW format을 의미하지 않는다.

### Reference Selection — CONFIRMED

- 공통 raw reference: `/home/binci/qwen_vision_test/photo_04.png`
- corresponding undistorted reference: `/home/binci/qwen_vision_test/photo_04_undistorted.png`
- legacy offline Homography: `/home/binci/qwen_vision_test/homography_12pt_undistorted.json`
- `photo_04.png`는 기존 Qwen single/multi bbox, 12-point collection/build, undistortion, grid와 Homography visualization의 공통 입력이다.
- 시각 확인상 `photo_04.png`에는 넓은 floor tile grid, 여러 물체와 사람/의자 등 주변 clutter가 있으나 predefined zone annotation은 없다.
- Homography JSON은 `photo_04_undistorted.png`, 2016×1133, undistorted pixel, meter 단위 X 0.0~1.2/Y 0.0~1.8을 기록한다.

### Geometry Compatibility — CONFIRMED WITH CONDITION

- Calibration 1920×1080을 source 2016×1133에 `scale_x=1.05`, `scale_y=1.049074074...`로 적용한다.
- `cv2.getOptimalNewCameraMatrix(..., alpha=1.0)` 후 `cv2.undistort()`/`cv2.undistortPoints()`를 사용한다.
- 저장된 `new_camera_matrix`를 사용하면 현재 OpenCV 4.5.4에서도 saved undistorted image와 point가 pixel-exact하게 일치한다.
- 현재 환경에서 `getOptimalNewCameraMatrix()`를 새로 계산하면 ROI `[110,186,1697,916]`으로 저장값 `[110,185,1700,918]`과 다르고 point는 최대 약 2.63 px 달라진다.
- 따라서 legacy offline pipeline은 JSON의 저장된 `new_camera_matrix`를 고정 입력으로 사용해야 한다.

### ArUco Presence — CONFIRMED TEST RESULT

- 15개 image 전체를 기존 `DICT_4X4_50` detector로 검사했으며 detected ID는 없었다.
- `photo_01.png`~`photo_08.png`에 정사각 fiducial 형태가 시각적으로 보이지만 configured detector가 인식하지 않았으므로 ArUco marker로 확정하지 않는다.

### Dataset Policy

- 기존 image/Homography는 `OFFLINE_DEVELOPMENT_DATASET`과 `LEGACY_OFFLINE_GEOMETRY`다.
- 추후 최종 camera pose에서 촬영할 `FINAL_PAPER_EXPERIMENTAL_DATASET`과 분리한다.
- legacy 12-point fitting error는 최종 논문 결과로 승격하지 않는다.

## Reference Inventory

### Camera Calibration — CONFIRMED

- `/home/binci/camera_calibration_1920x1080/calibration_result.json`
  - 1920×1080, inner corners 5×3, square 40 mm, selected frames 18
  - calibration RMS 0.205716 px
- 동등 형식: `camera_calibration_opencv.yaml`, `camera_calibration_ros2.yaml`
- `calibration_report.txt`: calibration이 원본 raw 영상이 아니라 제공된 screenshot에서 복원됐다고 명시
- `undistort_live.py`, `compare_raw_undistorted.py`: live 왜곡보정
- `aruco_pose.py`, `aruco_pose_id0.py`, `aruco_trajectory.py`: marker pose/trajectory
- `trajectory.csv`: 기존 측정 파일이나 생성 코드/version과 header provenance가 불명확

### Geometry and Qwen — CONFIRMED

- 기준 후보: `/home/binci/qwen_vision_test/homography_12pt_undistorted.json`
- 12점 적합 오차 재실행 결과: mean 5.34 mm, RMSE 5.74 mm, max 8.80 mm
- 같은 12점 leave-one-out 재실행 결과: mean 9.19 mm, RMSE 10.73 mm, max 19.94 mm
- 기준 image는 2016×1133 `photo_04.png`; calibration K는 1920×1080에서 비례 scaling됨
- `build_homography12_undistorted.py`, `check_homography12_undistorted_error.py`, `cross_validate_homography12_undistorted.py`
- `realtime_qwen.py`, 확장자 없는 `realtime_qwen`, `realtime_qwen_working.py`는 SHA-256이 동일한 중복본
- `qwen_photo04_bbox.py`, `qwen_photo04_multi_bbox.py`, `qwen_photo04_bbox_homography12.py`는 bbox→bottom-center→world 흐름의 부분 구현

**INFERRED:** undistorted 12-point homography가 현재 비교 자산 중 가장 좋은 후보지만, 5.34/5.74/8.80 mm는 학습에 사용한 점의 적합 오차라 독립 성능으로 사용할 수 없다.

**UNKNOWN:** 현재 카메라의 물리 설치가 homography 생성 시점과 동일한지, 2016×1133 입력 변환이 현재 capture pipeline과 같은지, Qwen model/cache와 GPU 환경이 현재 재현되는지 확인되지 않았다.

### Mecanum ROS2 — CONFIRMED

- Git commit: `3c710ab27209eaa196facb6a88507c42447837a8`
- tags: `v1.0-rpi-working`, `v2.0-jetson-working`, `v2.1-jetson-calibrated`, `v2.2-nav2-ready`
- packages: `mecanum_serial_hardware`, `mecanum_bringup`
- controller topic: `/mecanum_drive_controller/reference_unstamped`
- odometry: `/mecanum_drive_controller/odometry`; EKF output: `/odometry/filtered`
- wheel radius 0.04 m, center projection sum 0.199546 m, command timeout 0.3 s
- footprint `[[0.145,0.123],[0.145,-0.123],[-0.145,-0.123],[-0.145,0.123]]`, padding 0.01 m
- motion/IMU scripts와 Arduino firmware 존재
- forward/backward/left/right 10 cm의 `*_confirmed.py`는 이름 없는 대응본과 byte-identical; right 90° IMU도 동일. left 90° IMU 일반본과 confirmed본은 서로 다름.
- working tree에 기존 사용자 수정 4개가 있으므로 reference를 수정하지 않는다.

**UNKNOWN:** README의 Jetson hardware 성공 기록은 이번 조사에서 실제 재실행하지 않았다. Nav2 config는 존재하지만 external camera localization, map→odom, obstacle input과 full autonomous navigation은 문서상 미완성이다.

## Known Risks

1. calibration과 homography 생성 당시 camera 고정 상태가 현재와 다르면 homography 전체가 무효다. 기존 Homography는 `LEGACY_REFERENCE`이며 최종 논문 geometry가 아니다.
2. 1920×1080 calibration과 2016×1133 source 사이 scaling은 확인됐지만, `getOptimalNewCameraMatrix()` 재계산 결과가 OpenCV 환경에 따라 달라질 수 있다. Legacy offline 사용 시 저장된 `new_camera_matrix`를 사용한다.
3. ArUco marker length가 코드에 187 mm와 188 mm로 다르게 기록되어 있다.
4. robot marker center→robot reference transform과 workspace yaw 정의가 없다.
5. 기존 Qwen-world 코드 일부는 undistorted 기준이 아닌 `homography.json` 또는 `homography_12pt.json`을 사용한다.
6. 저장된 결과에 config/commit/hardware/run provenance가 일관되게 포함되지 않았다.

## STEP 21 — Stored-Image Geometry Audit + Offline Real-Scene Replay — PASS

- `/home/binci/qwen_vision_test`의 decodable image 15개를 inventory했다: `RAW_LIKE_SOURCE` 9, `UNDISTORTED_DERIVED` 1, `ANNOTATED_DERIVED` 2, `VISUALIZATION_DERIVED` 3.
- `photo_04.png`는 2016×1133 `LEGACY_GEOMETRY_REFERENCE / REFERENCE_VERIFIED`다. 저장 calibration, undistorted reference와 12-point Homography는 read-only로 감사했으며 metric transform은 실행하지 않았다.
- `photo_01/02/03/05/06/07/08.png`는 fixed-parameter ORB/RANSAC 진단에서 `DEVELOPMENT_COMPATIBLE_CANDIDATE`였다. inlier ratio 0.983386924–0.994142259, mean reprojection error 0.084511392–0.221448243 px, maximum reference-corner displacement 0.067397259–0.666570008 px였다.
- 640×480 `qwen_frame.jpg`는 `RESOLUTION_MISMATCH / NOT_ALLOWED`로 배제했다. Derived image는 replay 후보가 아니다.
- compatible group은 `photo_01.png`~`photo_08.png` 8개이며 순서는 `DETERMINISTIC_FILENAME_ORDER_NOT_REAL_CAPTURE_TIMELINE`이다. Dataset status는 `OFFLINE_REPLAY_READY`다.
- 기존 STEP 20 `SceneMonitor`를 변경 없이 재사용한 저장-image replay 결과는 `LOCAL_CHANGE` 7, `NO_SIGNIFICANT_CHANGE` 1, `GLOBAL_CHANGE` 0, semantic trigger 5였다. Tracking continuity는 사진 사이 실제 시간 간격을 알 수 없어 `DEVELOPMENT_ONLY`다.
- Audit와 replay는 즉시 반복에서 동일했고 system/`qwen_vl` Python 모두 349/349 tests가 PASS했다. STEP 12–20 artifacts는 변경되지 않았다.
- `OFFLINE STORED-IMAGE REPLAY IS NOT LIVE REAL-TIME EVIDENCE`. Qwen, world coordinate, physical obstacle footprint, planner injection, ROS/Nav2와 robot execution은 수행하지 않았다.

## STEP 22 — Offline Observation → VLM Semantic + Metric Point Grounding — SOFTWARE PASS / PIPELINE PARTIAL

- STEP 22 connects triggered pixel observations to VLM semantics and metric point grounding only.
- STEP 21 compatible replay 8개 중 STEP 20 `semantic_trigger=true`인 5개 image와 confirmed candidate 58개만 actual Qwen 입력으로 사용했다.
- Local `Qwen/Qwen3-VL-4B-Instruct`를 CPU에서 8회 실행했다: primary 5개 frame, `photo_02/03` repeat 2회씩, `photo_06` strict-prompt retry 1회. 모든 raw text와 parsed frame result를 저장했다.
- Primary 결과는 `NO_VISIBLE_ENTITY` 46, `VISIBLE_ENTITY_FOUND` 1, `AMBIGUOUS` 0이었다. Visible 결과는 association되지 않았고 valid metric point는 0개였다.
- `photo_02`는 candidate bbox copy mismatch로 2회 모두 거부됐다. `photo_06` V1의 visible bbox는 0–1 domain이었고, V2는 raw pixel 값을 normalized field에 복사해 0–1000 범위를 위반했다. 좌표를 임의 확대하거나 association threshold를 완화하지 않았다.
- Software와 36개 신규 검증은 PASS지만 실제 semantic metric point가 0개이므로 `semantic_metric_pipeline=PARTIAL`, STEP result는 `PARTIAL`이다.
- Exhaustive mode는 종료하고 original 2016×1133 source image와 candidate 1개만 사용하는 representative validation으로 축소했다. 기존 exhaustive directory의 73개 파일은 content digest `1de4c7cb...a32d9`로 변경 없이 보존됐다.
- Representative attempt는 최대 3개 정책을 지켰다: `photo_06/track_011` person candidate, `photo_08/track_039` tennis-racket candidate, `photo_07/track_036` paper-cup candidate. 세 actual CPU run 모두 strict parse는 PASS였지만 `NO_VISIBLE_ENTITY`였고 association과 LegacyGeometry는 `NOT_EVALUATED/NOT_RUN`이었다.
- Representative valid metric point는 0개다. Software integration, actual inference, strict gates는 PASS이며 semantic-to-metric representative integration과 STEP 22 overall은 PARTIAL이다. 추가 Qwen inference는 중지했다.
- Scene-change candidate geometry is never used directly as physical obstacle geometry.
- Metric grounding uses the bottom-center of a validated VLM visible-object bbox through the existing deterministic legacy geometry pipeline.
- A metric point is not a physical obstacle footprint.
- Planner injection remains prohibited until obstacle footprint geometry has an explicit source.

## STEP 22B — Candidate-Conditioned Observation Audit — PASS

- STEP 22 failed to produce a representative metric point because the semantic candidate-conditioning gate did not yield a validated visible-object bbox.
- Candidate coordinates from STEP 20 are raw image pixels, while Qwen object bbox output uses normalized 0–1000 coordinates.
- SceneMonitor code constructs `[x,y,x+width,y+height]`, confirming half-open `[x1,y1,x2,y2)` boxes. Therefore `photo_06`의 `x2=2016`은 2016-pixel-wide image에서 valid right-boundary touch다.
- Actual representative prompt는 raw-pixel metadata라고 명시했지만 image width/height, `[x_min,y_min,x_max,y_max]`, half-open/exclusive convention을 포함하지 않았다. 세 prompt의 coordinate semantics는 `LIKELY_CONFUSING`으로 분류했다.
- Audit-derived normalized candidates는 person `[709,196,1000,964]`, racket `[504,481,682,604]`, cup `[347,398,370,457]`이다. 기존 Qwen output 보정이나 metric grounding에는 사용하지 않았다.
- Visual audit에서 person candidate는 `OBJECT_PLUS_BACKGROUND`, racket/cup은 `FULL_OBJECT_LIKE`였다. Cup bbox는 full image의 약 0.135%로 작다.
- Candidate-conditioning representation is audited before any further model inference. Qwen model load/generation은 0회였고 기존 raw output 11개의 전후 digest가 동일했다.
- Scene-change candidate geometry remains separate from full-object semantic geometry.
- STEP 22C prototype으로 same-resolution full-scene overlay와 normalized 0–1000 candidate metadata를 결합하는 Method C를 권고한다. 실제 inference는 수행하지 않았다.

## STEP 22C — Candidate-Conditioned VLM Grounding Revalidation — PARTIAL

- `photo_08 / image_014_photo_08 / track_039` 한 사례에 STEP 22B Method C를 적용해 actual Qwen inference를 정확히 1회 실행했다. 다른 candidate, Method D와 retry는 실행하지 않았다.
- STEP 22C uses candidate overlay and normalized candidate metadata solely for semantic attention.
- 모델 입력은 기존 2016×1133 full-scene highlight overlay이고 candidate metadata는 `[504,481,682,604]` normalized integer 0–1000만 제공했다. Raw `[1017,545,1375,684]`는 prompt에서 제외하고 provenance/association에만 보존했다.
- Actual raw output은 strict parser를 통과한 `NO_VISIBLE_ENTITY`였다. Full-object bbox가 없으므로 association은 `NOT_EVALUATED`, bottom-center/LegacyGeometry/world XY는 `NOT_RUN`이다.
- Metric coordinates are derived only from a validated full-object bbox mapped back to the original full-image domain.
- Overlay pixels are never used as Homography input.
- One representative actual inference was executed.
- STEP 22C와 최종 STEP 22 semantic-metric integration은 `PARTIAL`이다. Physical geometry는 `NOT_AVAILABLE`, planning role은 `NOT_ASSIGNED`, planner injection은 `NOT_RUN`이다.
- System Python과 `qwen_vl` Python 사전 회귀는 각각 425/425 PASS였고, 기존 보호 artifact digest와 원본 image hash는 실행 전후 동일했다.

## STEP 22D — Full Image + Secondary Crop Candidate Conditioning Revalidation — PASS

- STEP 22C와 동일한 `photo_08 / image_014_photo_08 / track_039`에 Method D actual Qwen inference를 정확히 1회 실행했다. 다른 candidate, crop margin, retry와 fallback은 없었다.
- Method D uses a secondary crop only to improve semantic visibility. IMAGE 1은 원본 2016×1133 full scene이고 IMAGE 2는 STEP 22B의 466×181, 15% context crop이다.
- Qwen은 `VISIBLE_ENTITY_FOUND / tennis racket`, `bbox_reference_image=IMAGE_1`, full-image normalized bbox `[504,481,682,604]`를 반환했고 strict parser가 수용했다.
- Full-image raw bbox `[1016.064,544.973,1374.912,684.332]`와 STEP 20 candidate의 association은 `ASSOCIATED`, IoU `0.9945790206`이었다.
- Bottom-center `[1195.488,684.332]` raw px를 stored undistortion `[1098.0327,752.1130]` px와 existing legacy Homography에 통과시켜 `LEGACY_WORKSPACE_COORDINATE [0.764672,0.713486] m`, `INSIDE`를 생성했다.
- All metric coordinates continue to originate from a full-object bbox expressed in the original full-image coordinate system. Crop coordinates are never mapped through Homography.
- STEP 22D executes exactly one actual Qwen inference. 최초 deterministic visualization 후처리에서 기존 track-ID lookup 오류가 발생했으나 raw output은 이미 보존됐고, 추가 inference 없이 helper를 수정해 후처리만 완료했다. 따라서 inference timing은 복구하지 않고 `NOT_RECOVERED`로 기록했다.
- STEP 22D와 final semantic-metric integration STEP 22는 `PASS`로 승격한다. Physical geometry는 `NOT_AVAILABLE`, planning role은 `NOT_ASSIGNED`, planner injection은 `NOT_RUN`이다.

## STEP 23 — Metric Obstacle Footprint Source + Ground-Contact Geometry — PASS

- STEP 22D의 `photo_08 / track_039 / tennis racket` semantic metric point `[0.764672,0.713486] m`를 physical-geometry layer에 읽기 전용으로 연결했다.
- `/home/binci/vlm_ws`와 `/home/binci/qwen_vision_test` 감사에서 tennis-racket의 실측 length, width, ground footprint 또는 bottom-center-to-footprint anchor 근거를 찾지 못했다. 실제 record는 `geometry_source=UNKNOWN`, `physical_footprint=NOT_AVAILABLE`, `representative_physical_geometry=NOT_READY_MEASUREMENT_REQUIRED`다.
- A semantic metric point is not itself a physical obstacle footprint. Physical object geometry requires an explicit dimensional source. The VLM bounding box is not used to infer metric width or length. Observed bottom-center is not automatically the physical footprint center.
- Physical geometry, uncertainty, safety margin, and planner representation are separate layers. Planning geometry는 `NOT_GENERATED`, planner injection은 `NOT_RUN`이다.
- 기존 STEP 14 `dev_obstacle_01`만 `STEP14_EXISTING_DEVELOPMENT_SYNTHETIC` provenance로 재사용해 0.20×0.50 m local rectangle의 world translation을 검증했다. 이것은 실제 tennis-racket geometry가 아니다.
- Qwen inference 0회, STEP 17/18/19·A*·costmap 호출 0회다. 보호 대상 453개 artifact digest는 전후 `10e68938...bfd8`로 동일했다.
- 신규 26개를 포함한 전체 회귀는 system 및 `qwen_vl` Python에서 각각 471/471 PASS했다.

## STEP 23M — Representative Object Physical Measurement Registration — PASS / WAITING

- `/home/binci/vlm_ws`, `/home/binci/qwen_vision_test`와 STEP 23 config/results를 다시 감사했지만 `USER_PHYSICAL_MEASUREMENT`로 명시된 tennis-racket 치수나 anchor evidence는 없었다. Actual input `config/representative_object_measurement.json`은 생성하지 않았다.
- Null-only template과 READY-only empty registry를 추가했다. Shape는 사용자가 `CIRCLE`, `AXIS_ALIGNED_RECTANGLE`, `POLYGON` 중 직접 선택해야 하며 semantic label로 자동 선택하지 않는다.
- Intake validator는 `mm`, `cm`, `m` raw input을 canonical meter로 변환하고 raw/canonical provenance를 모두 보존한다. Positive rectangle/circle dimensions, finite non-degenerate polygon, measured anchor와 필요한 orientation이 모두 있어야 `READY`다.
- 현재 상태는 `WAITING_FOR_USER_MEASUREMENT`; registry object 0개, physical world placement `NOT_RUN`, planning geometry `NOT_GENERATED`, planner injection `NOT_RUN`, Qwen inference 0회다.
- 보호 대상 461개 STEP 12–23 artifact digest는 전후 `f1d56b83...e1c1`로 동일했다. 신규 22개를 포함한 전체 회귀는 system 및 `qwen_vl` Python에서 각각 493/493 PASS했다.

## STEP 24N — Approximate Navigation Obstacle Adapter — PASS

- Preserved STEP 22D `image_014_photo_08 / track_039 / tennis racket` evidence was imported without Qwen inference or Homography recomputation. The validated reference point remains `[0.764672, 0.713486] m` in `LEGACY_WORKSPACE_COORDINATE`.
- Semantic `VISIBLE_ENTITY_FOUND`, association `ASSOCIATED`, metric `AVAILABLE`, frame, finite-coordinate, allowed-provenance and workspace `INSIDE` gates are mandatory. The navigation role is `OBSTACLE_CANDIDATE`; `track_039` remains temporal observation identity rather than semantic identity.
- The existing STEP 15 `metric_to_grid` helper and configured 0.02 m resolution project the point to row 35, column 38 with cell center `[0.77, 0.71] m`. This `SINGLE_OCCUPIED_CELL` is navigation discretization, not physical object size; the stored grid was not modified.
- Physical geometry and navigation obstacle approximation are separate layers. STEP 23M remains `WAITING_FOR_USER_MEASUREMENT / NOT_READY`, but this does not block the `NAVIGATION_APPROXIMATION`. The bbox bottom-center is not claimed as an object or footprint center, and no bbox/label-derived size was generated.
- STEP 17 accepts only development-synthetic axis-aligned rectangles, so direct dynamic-world insertion is unsupported until STEP 25N. STEP 17/18/19, A*, costmap, ROS/Nav2 and robot execution were all `NOT_RUN`.

## STEP 25N — Navigation Dynamic-World Integration — PASS

- STEP 24N `image_014_photo_08_track_039`을 source of truth로 읽고 exact 0.02 m containing-cell bounds `[0.76,0.78]×[0.70,0.72] m`를 STEP 17 axis-aligned rectangle contract에 연결했다.
- Record는 `NAVIGATION_APPROXIMATION / NAVIGATION_DISCRETIZATION_CELL / STEP24N_NAVIGATION_APPROXIMATION`이며 `DEVELOPMENT_SYNTHETIC` 또는 physical footprint로 재분류하지 않았다. Reference point, row/column, bounds, source image와 `track_039` provenance가 모두 유지된다.
- Deterministic observation-scoped ID `dynamic_image_014_photo_08_track_039`를 사용해 기존 `apply_obstacle_event`로 ADD했고 world revision은 0→1이 됐다. Track identity는 `OFFLINE_TEMPORAL_OBSERVATION`이며 persistent physical identity를 주장하지 않는다.
- Source cell occupancy는 기존 static collision 때문에 1→1로 동일하다. 하지만 existing robot-footprint collision transform을 cell rectangle에 적용한 결과 주변 78개 cell이 0→1로 변경됐다. 이는 object-size inference나 신규 inflation이 아니다.
- Static obstacle state와 보호 대상 471개 이전 artifact는 불변이다. STEP 17 collision world/grid rebuild만 수행했고 path validation, STEP 18/19, A*, planning, replanning, ROS/Nav2는 `NOT_RUN`이다.

## STEP 26N — Dynamic Observation → Existing Path Validity — PASS

- STEP 25N before/after world에 기존 STEP 17 D1의 저장된 90-waypoint `SAFETY_COST_ASTAR` current path를 그대로 적용했다. Path를 생성·수정하지 않았으며 전후 SHA-256는 모두 `17646cb8...dcdc7`이다.
- Existing `validate_current_path`를 재사용했다. 이 helper는 waypoint workspace/collision/grid, continuous segment/AABB intersection, 8-connected transition과 diagonal corner-cut을 검사하고 first blocked waypoint/edge 및 obstacle ID를 반환할 수 있다.
- Representative result는 revision 0 `VALID` → revision 1 `VALID`, classification `NO_PATH_EFFECT`다. 따라서 blocking location과 dynamic-obstacle attribution은 `NOT_APPLICABLE_AFTER_PATH_VALID`이다.
- Source cell은 이미 occupied였고 STEP 25N에서 robot-collision grid 78개 cell이 추가됐지만 canonical D1 path와 교차하지 않았다. 이 evidence를 변경해 block을 제조하지 않았다.
- STEP 23M은 `WAITING_FOR_USER_MEASUREMENT` 그대로다. STEP 18/19, A*, new path generation, replanning, ROS/Nav2는 `NOT_RUN`이다.

## STEP 27N — Controlled Observation–Path Effect Experiment — PASS

- CASE A는 STEP 26N observed navigation approximation을 읽기 전용 control로 보존했으며 동일 path SHA-256 `17646cb8...dcdc7`에서 `VALID → VALID / NO_PATH_EFFECT`다.
- Existing STEP 17 fixture audit에서 documented D2 `D2_add_on_path`를 CASE B로 선택했다. 저장된 path/config/event를 수정하지 않고 기존 state, collision-world, grid, validity API로 재현한 결과 `VALID → BLOCKED / PATH_BLOCKED`다.
- CASE B 최초 차단은 waypoint 50, edge 49, grid `[36,46]`, world `[0.93,0.73] m`이며 cell occupancy `0→1`, blocker는 `dynamic_box_02`다.
- CASE A는 `OBSERVED_NAVIGATION_APPROXIMATION`, CASE B는 `DOCUMENTED_DEVELOPMENT_FIXTURE`로 명시적으로 분리했다. CASE B를 camera/VLM 또는 실제 관측 evidence로 재분류하지 않았다.
- Physical geometry는 `NOT_READY_NOT_USED`, STEP 23M은 `WAITING_FOR_USER_MEASUREMENT`다. Qwen, Homography, STEP 18/19, A*, new path generation, replanning, ROS/Nav2는 모두 `NOT_RUN`이다.
- STEP 26N 9개 source artifact와 보호 대상 487개 artifact digest는 전후 동일했다. 신규 28개를 포함한 전체 회귀는 system 및 `qwen_vl` Python에서 각각 617/617 PASS했다.

## STEP 28N — Path-Validity BLOCKED → Existing Persistence / STOP-WAIT Policy Integration — PASS

- STEP 27N path-validity evidence를 기존 STEP 19 정책 입력에 연결했다. `PATH_BLOCKED`, `PERSISTENT_BLOCKING`, `REPLAN_REQUIRED`, replanning execution은 서로 다른 상태다.
- CASE A `OBSERVED_NAVIGATION_APPROXIMATION / VALID`은 dynamic obstacle이 존재해도 `MOVING / CONTINUE_CURRENT_PATH`를 유지했으며 STOP/WAIT을 유발하지 않았다.
- CASE B는 `DOCUMENTED_DEVELOPMENT_FIXTURE`인 STEP 17 D2 `dynamic_box_02`다. STEP 19의 기존 P2 logical-time sequence를 재사용했으며 live camera temporal evidence로 취급하지 않았다.
- Existing parameters를 변경하지 않았다: presence 2, absence 2, interval 0.2 s, persistence threshold 1.0 s. 0.2 s에 presence confirm/`STOPPED_FOR_OBSTACLE`, 이후 `WAITING_FOR_CLEARANCE`, blocking duration 1.0 s에 `PERSISTENT_BLOCKING / REPLAN_REQUIRED`가 결정됐다.
- `REPLAN_REQUIRED`는 decision output만 의미한다. `execute_replanning=False`로 STEP 18은 `NOT_RUN`, planner runs 0, A*와 new path generation은 `NOT_RUN`, active path revision은 0이다.
- STOP/WAIT은 offline policy state이며 physical robot stop을 검증하지 않는다. CASE B disappearance는 만들지 않았고 RESUME는 기존 STEP 19 P1 read-only contract evidence만 사용했다.
- STEP 23M은 `WAITING_FOR_USER_MEASUREMENT`, physical geometry는 `NOT_READY_NOT_USED`다. STEP 12–27N 보호 artifact는 변경되지 않았다.
- Focused tests 32/32 PASS. 전체 회귀는 system Python 649/649 PASS (7.199 s), `qwen_vl` 649/649 PASS (7.064 s)다.

## STEP 29N — Confirmed Persistent Block → Existing STEP 18 Replanning Integration — PASS

- STEP 28N의 decision-only boundary를 그대로 가져와 `BLOCKED + PERSISTENT_BLOCKING + REPLAN_REQUIRED + valid provenance`일 때만 기존 STEP 18 `evaluate_event_and_replan`을 호출하는 최소 gate를 추가했다.
- CASE A `OBSERVED_NAVIGATION_APPROXIMATION / VALID / CONTINUE_CURRENT_PATH`는 STEP 18 호출 0회, planner run 0회, new path 없음, revision 0 유지다. Obstacle presence alone does not trigger replanning.
- CASE B는 camera/VLM evidence가 아닌 `DOCUMENTED_DEVELOPMENT_FIXTURE`다. STEP 17 D2의 `dynamic_box_02`, world revision 1, 저장 path, STEP 28N artifact에서 읽은 waypoint 20, 기존 goal/config를 변경 없이 사용했다.
- CASE B에서 기존 STEP 18을 정확히 1회 호출했고 existing `SAFETY_COST_ASTAR`가 `REPLAN_SUCCESS / SWITCH_TO_REPLANNED_PATH`를 반환했다. Old path는 90 waypoints, 1.8462741699796958 m, revision 0에서 `SUPERSEDED`; replacement는 99 waypoints, 2.092548339959391 m, revision 1이다.
- Existing `validate_replanned_path`는 path validity, current-pose start cell, first waypoint, original-goal cell과 final waypoint를 모두 통과해 `VALID`을 반환했다. 이 결과는 offline software integration evidence이며 camera-derived end-to-end replanning이나 physical robot execution이 아니다.
- STEP 18의 기존 failure contract도 historical R4 artifact로 보존 확인했다: `REPLAN_FAILED / STOP`, replacement 없음. Planner config와 STEP 28N 9개 artifact, 이전 results 500개는 불변이다.
- STEP 23M은 `WAITING_FOR_USER_MEASUREMENT`, physical geometry는 `NOT_READY_NOT_INFERRED`다. Qwen/Homography/ROS/Nav2/hardware execution은 모두 `NOT_RUN`이다.
- Focused tests 32/32 PASS. 전체 회귀는 system Python 681/681 PASS (11.760 s), `qwen_vl` 681/681 PASS (11.173 s)다.

## STEP 30N — Offline Navigation Chain System Integration — PASS

- 기존 STEP22D–29N artifact만 읽는 system-level audit를 추가했다. Qwen, Homography, dynamic-world rebuild, path validation, persistence replay와 STEP18을 다시 실행하지 않았다.
- TRACK A `OBSERVED_NAVIGATION_APPROXIMATION`은 actual `image_014_photo_08 / track_039 / tennis racket`에서 world reference `[0.764672,0.713486] m`, navigation cell `(35,38)`, dynamic ID `dynamic_image_014_photo_08_track_039`까지 identity를 유지했다. 결과는 `VALID / NO_PATH_EFFECT / CONTINUE_CURRENT_PATH`, STEP18 invocation 0, replacement 없음이다.
- TRACK B는 별도 `DOCUMENTED_DEVELOPMENT_FIXTURE`다. STEP17 D2 `dynamic_box_02`에서 `BLOCKED → PERSISTENT_BLOCKING → REPLAN_REQUIRED → REPLAN_SUCCESS → replacement VALID` lineage와 revision 0→1을 보존 확인했다.
- 두 track은 결합되지 않았다. 현재 지원되는 주장은 actual camera/VLM observation이 navigation에 들어가 no path effect를 올바르게 처리했고, 별도 development blocker가 downstream block/persistence/replanning을 검증했다는 것이다. Observed tennis racket이 replanning을 유발했다는 주장은 지원되지 않는다.
- Approximate navigation observation에는 physical dimension이 필수는 아니지만, STEP23M은 계속 `WAITING_FOR_USER_MEASUREMENT`, physical geometry는 `NOT_READY`다. 실제 footprint/collision accuracy/manipulation readiness는 주장하지 않는다.
- 이전 result artifact 510개 digest는 전후 `3d64c51c...0606034`로 동일했다. Qwen inference 0회, Homography/ROS/Nav2/hardware execution은 `NOT_RUN`이다.
- Focused tests 24/24 PASS. 최종 전체 회귀는 system Python 705/705 PASS (12.007 s), `qwen_vl` 705/705 PASS (11.303 s)다.
