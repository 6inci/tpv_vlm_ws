# Decisions

## D001 — Source of Truth

- **Decision:** `/home/binci/vlm_ws`만 현재 연구 개발의 Source of Truth로 사용한다.
- **Reason:** 기존 세 저장소는 서로 다른 실험 시점의 reference이고 중복/부분 구현을 포함한다.
- **Consequence:** 필요한 최소 자산만 provenance와 함께 port한다. 전체 repository는 복사하지 않는다.
- **Status:** ACCEPTED (2026-09-16).

## D002 — Geometry Baseline

- **Decision:** `homography_12pt_undistorted.json`은 `LEGACY_REFERENCE`로만 보존하고 최종 논문 Homography로 사용하지 않는다. 최종 camera pose를 고정한 뒤 native 1920×1080 image로 새 Homography를 생성한다.
- **Offline use:** `photo_04_undistorted.png`와 JSON에 저장된 `new_camera_matrix`를 함께 사용하는 경우 `LEGACY_OFFLINE_GEOMETRY`로 허용한다. 런타임에서 `getOptimalNewCameraMatrix()`를 재계산하지 않는다.
- **Evidence:** 저장된 12점 적합 오차를 재실행했을 때 mean 5.34 mm, RMSE 5.74 mm, max 8.80 mm였다. LOO는 mean 9.19 mm, RMSE 10.73 mm, max 19.94 mm였다.
- **Limitation:** 두 값 모두 같은 12점 집합에 기반하며 독립 floor-point 성능이 아니다. Calibration은 1920×1080이지만 source image는 2016×1133이고 향후 camera pose도 변경된다.
- **Rejected for current baseline:** distorted `homography_12pt.json`과 ray-plane 후보. 현재 기록상 undistorted homography보다 우선할 독립 증거가 없다.
- **Status:** SUPERSEDED 2026-09-17 — legacy 수치는 논문 최종 성능으로 인용하지 않는다.

## D003 — VLM and Geometry Separation

- **Decision:** VLM은 semantic label/bbox만 생성하고 metric X/Y는 calibration/homography 코드가 계산한다.
- **Reason:** 모델 출력을 불확실 입력으로 격리하고 좌표 변환을 재현 가능하게 한다.
- **Status:** ACCEPTED.

## D004 — Front Gripper Instead of Push

- **Decision:** object handling baseline은 front-mounted simple 1-DOF gripper/lock으로 제한한다.
- **Reason:** grasp 상태를 명시하고 운반 중 object-relative footprint를 정의할 수 있으며 범용 manipulation 없이 연구질문을 검증할 수 있다.
- **Status:** DESIGN; Phase 0/8에서 실제 hardware를 확정한다.

## D005 — Deterministic 2D Planner

- **Decision:** 초기 planning baseline은 occupancy grid + A*로 한다. 기존 Nav2는 필요할 때 실행 계층으로 재사용한다.
- **Reason:** planner 자체가 contribution이 아니며 두 footprint의 collision-free 결과를 단순하게 검증할 수 있다.
- **Evidence:** 기존 Nav2 config가 있으나 `NavfnPlanner use_astar: false`이고 external localization/costmap 통합은 미완성이다.
- **Status:** ACCEPTED FOR DESIGN; Phase 11에서 검증한다.

## D006 — MCP and Unrelated Extensions Excluded

- **Decision:** MCP, 새 SLAM/Nav2/ArUco 알고리즘, VLM fine-tuning, RL, dynamic obstacles를 현재 논문에 추가하지 않는다.
- **Reason:** 핵심 질문인 semantic→metric→physical repositioning의 실험 근거를 강화하지 않고 일정과 변수만 늘린다.
- **Status:** FROZEN SCOPE.

## D007 — Reference Repositories Remain Read-Only

- **Decision:** 세 reference 경로의 dirty/working 상태를 그대로 보존한다.
- **Evidence:** `mecanum_ros2`에는 조사 시점에 tracked 수정 3개와 untracked 파일 1개가 있었다.
- **Status:** ACCEPTED.

## D008 — Early Development Uses Still Images

- **Decision:** Geometry, ArUco, VLM, workspace와 path-planning 초기 개발 입력은 기존 pre-captured still image를 사용한다.
- **Development dataset:** 기존 image는 `OFFLINE_DEVELOPMENT_DATASET`이며 최종 논문 dataset과 분리한다.
- **Final target:** 최종 camera pose가 고정된 뒤 native 1920×1080 color image로 `FINAL_PAPER_EXPERIMENTAL_DATASET`을 새로 촬영한다.
- **Reason:** 동일 입력을 반복 사용해 알고리즘 결과를 재현 가능하게 비교한다.
- **Deferred:** continuous streaming과 live/periodic feedback은 robot execution 단계까지 구현하지 않는다.
- **Status:** ACCEPTED (2026-09-17).

## D009 — Existing Calibration Recommendation

- **Decision:** 현재 권고는 `RECALIBRATION_RECOMMENDED`다. 사용자 승인 전에는 실행하지 않는다.
- **Reason:** 외장 camera가 검출되지 않아 기존 `Ucamera001`과의 물리 identity를 확인할 수 없고, 기존 보고서는 원본 raw frame이 아닌 screenshot에서 복원됐다고 명시한다.
- **Reconsideration:** 동일 physical camera/lens/focus와 native 1920×1080 조건이 직접 확인되면 `REUSE_CANDIDATE`로 재평가할 수 있다.
- **Status:** PROVISIONAL (2026-09-17).

## D010 — Offline Reference Image

- **Decision:** 공통 raw reference는 `/home/binci/qwen_vision_test/photo_04.png`, 대응 undistorted reference는 `photo_04_undistorted.png`로 고정한다.
- **Reason:** 기존 Qwen bbox, 12-point Homography, point collection, undistortion과 visualization이 모두 `photo_04.png`를 기준으로 연결된다.
- **ArUco:** 기존 `DICT_4X4_50` detector는 reference image를 포함한 15개 image에서 marker를 검출하지 못했으므로 별도 ArUco reference는 지정하지 않는다.
- **Status:** ACCEPTED FOR OFFLINE DEVELOPMENT (2026-09-17).

## D011 — Legacy Geometry API Contract

- **Decision:** offline legacy 변환은 `raw_pixel_to_undistorted_pixel`, `undistorted_pixel_to_world`, `raw_pixel_to_world`의 세 단계 API로 제공하고 raw/undistorted 좌표계를 섞지 않는다.
- **Configuration:** calibration intrinsic/distortion은 `calibration_result.json`, output camera matrix와 Homography는 `homography_12pt_undistorted.json`의 저장값을 사용한다. 런타임 `getOptimalNewCameraMatrix()` 재계산은 금지한다.
- **Validation:** 입력 image resolution은 legacy 2016×1133과 일치해야 하며 불일치하면 변환을 거부한다.
- **Evidence:** 2026-09-17 software regression에서 reference image와 12점 변환을 정확히 재현했고 반복 실행 산출물도 byte-identical했다.
- **Limitation:** API의 좌표계는 물리 원점/축 의미가 확인되지 않은 `LEGACY_WORKSPACE_COORDINATE`이며 final-paper geometry가 아니다.
- **Status:** ACCEPTED FOR OFFLINE DEVELOPMENT (2026-09-17).

## D012 — Single-Object Qwen Grounding Contract

- **Decision:** Qwen은 semantic label과 0–1000 normalized `bbox_2d`만 출력하고 metric 좌표는 계산하지 않는다.
- **Conversion:** normalized bbox를 2016×1133 raw pixel bbox로 변환한 뒤 bottom-center를 object reference pixel로 사용하고, D011의 geometry API로 undistorted pixel과 `LEGACY_WORKSPACE_COORDINATE` X/Y를 계산한다.
- **Validation:** bbox 누락, JSON/schema 오류, 범위 밖 좌표, 역전된 bbox와 다중 후보는 조용히 보정하거나 첫 항목을 선택하지 않고 오류로 처리한다.
- **Runtime:** 기존 `qwen_vl` environment와 local-only `Qwen/Qwen3-VL-4B-Instruct` cache를 사용하며 자동 설치·upgrade·download를 하지 않는다.
- **Evidence:** `photo_04.png` red-cup fresh inference 3회에서 동일 bbox와 동일 world X/Y를 얻었고 저장 legacy bbox integration도 PASS했다.
- **Limitation:** bottom-center는 floor contact의 근사이며 object position ground-truth가 없어 최종 정확도로 해석하지 않는다.
- **Status:** ACCEPTED FOR OFFLINE DEVELOPMENT (2026-09-17).

## D013 — Multi-Entity Scene Grounding Contract

- **Decision:** multi-entity model output은 `label`, 제한된 `semantic_type`, 제한된 `navigation_role`, normalized `bbox_2d`만 포함한다. 허용 vocabulary 밖 값은 `unknown`으로 조용히 바꾸지 않고 validation failure로 처리한다.
- **Entity ID:** Qwen이 ID를 만들지 않으며 parse된 순서에 따라 Python이 image-local `entity_001` 형식으로 할당한다. Persistent tracking ID가 아니다.
- **Geometry:** STEP 05 bbox 변환과 STEP 04 geometry를 재사용한다. bottom-center world X/Y가 configured legacy bounds X 0.0–1.2 m, Y 0.0–1.8 m 밖이면 `OUTSIDE_LEGACY_WORKSPACE`로 표시하고 clamp하지 않는다.
- **Repeatability:** entity 순서만 신뢰하지 않고 normalized label과 bbox IoU 0.5 이상을 이용해 run 간 entity를 분석용으로 matching한다. IoU는 accuracy가 아니라 run-to-run consistency다.
- **Obstacle limit:** bbox 전체를 floor Homography로 변환하지 않으며 obstacle polygon, physical footprint와 occupancy grid를 만들지 않는다.
- **Evidence:** 3회 모두 11개 entity/common entity를 재현했지만 8개가 workspace 밖이고 실제 물체 label이 generic하여 integration 결과를 PARTIAL로 판정했다.
- **Status:** ACCEPTED WITH REWORK REQUIRED (2026-09-17).

## D014 — Semantic Identity and Task/Planning Role Separation

- **Decision:** VLM perception output은 specific visible `label`, 제한된 `semantic_type`, normalized `bbox_2d`까지만 담당한다. STEP 06의 model-generated `navigation_role`은 deprecated하며 새 semantic scene parser에서 보존하거나 신뢰하지 않는다.
- **Responsibility:** task layer가 instruction에 따라 target/destination을 선택하고, workspace/validation layer가 obstacle/planning role을 결정하며, geometry만 pixel을 metric coordinate로 변환한다.
- **Reference scope:** `/home/binci/vlm_ws/datasets/offline_dev/annotations/photo_04_semantic_reference.json`은 legacy record에서 확인된 세 object만 포함하는 `PARTIAL_SEMANTIC_REFERENCE`다. 다른 entity는 `UNREVIEWED`이며 ground truth로 승격하지 않는다.
- **Label policy:** ontology word를 label로 쓴 경우 `GENERIC`, `unknown`은 `UNKNOWN`, 그 외 visible name은 `SPECIFIC`으로 표시한다. Reference semantic PASS는 bbox IoU association과 reviewed alias match를 모두 요구한다.
- **Active workspace:** geometry `VALID/OUTSIDE/INVALID`를 `INSIDE/OUTSIDE/INVALID_GEOMETRY`로 분리한다. OUTSIDE entity는 scene에 보존하며 planner input filtering은 아직 구현하지 않는다.
- **Evidence:** 새 prompt는 3회 모두 generic label을 제거했지만 reference PASS는 trophy 1개뿐이어서 semantic refinement를 PARTIAL로 판정했다.
- **Status:** ACCEPTED WITH REWORK REQUIRED (2026-09-17).

## D015 — Scene Inventory and Task-Conditioned Target Grounding

- **Decision:** Broad Scene Inventory와 Task-Conditioned Target Grounding은 목적과 성공 조건이 다른 perception mode로 분리한다. Broad scene output은 모든 가능한 task target을 빠짐없이 열거해야 할 의무가 없다.
- **Task query:** task layer가 선택한 canonical target phrase만 VLM prompt에 제공한다. Reference bbox, pixel, world coordinate는 제공하지 않는다. 현재는 partial reference label을 query source로 사용하며 full natural-language task parser는 아직 없다.
- **Candidate rule:** 0개 후보는 `TARGET_NOT_FOUND`, 2개 이상은 `TARGET_AMBIGUOUS`이며 임의의 첫 후보를 좌표화하지 않는다. 정확히 한 후보만 STEP 05 bbox pipeline과 STEP 04 geometry로 변환한다.
- **Verification:** Reference IoU 0.5 이상은 `REFERENCE_POSITION_MATCH`라는 association 근거일 뿐 accuracy threshold가 아니다. Semantic label/alias status와 position status는 서로 독립적으로 기록한다.
- **Alias policy:** Position이 일치한 새로운 specific label은 `HUMAN_ALIAS_CANDIDATE`일 뿐 reference alias에 자동 추가하지 않는다. 따라서 broad scene의 `red mug`는 자동 alias가 아니며, task-conditioned `red cup` 출력으로 alias 목록을 변경하지 않았다.
- **Workspace:** world point는 configured bounds에 따라 `INSIDE`, `OUTSIDE`, `INVALID_GEOMETRY`로 표시하고 clamp하지 않는다. `OUTSIDE` target은 검출 결과에는 보존하되 이후 physical feasibility 검증 전 실행 대상으로 간주하지 않는다.
- **Evidence:** 세 target 각 3회가 모두 position match였고 broad scene에서 누락된 white bag도 task mode에서 3/3회 복구됐다.
- **Status:** ACCEPTED FOR OFFLINE DEVELOPMENT (2026-09-17).

## D016 — Deterministic Metric Goal Zones

- **Decision:** Offline destination zone은 VLM이 생성하는 metric region이 아니라 workspace configuration에 저장된 deterministic metric polygon이다.
- **Resolution flow:** VLM/LLM 또는 deterministic parser는 semantic destination name을 `zone_A` 같은 ID로 resolve할 수 있지만 실제 polygon과 center는 `/home/binci/vlm_ws/config/zones.json`에서만 가져온다. Model이 meter coordinate를 직접 생성하지 않는다.
- **Development scope:** 현재 여섯 zone은 legacy 0.6 m grid cell을 재사용한 `OFFLINE_DEVELOPMENT_ZONE_SET`이다. Physical marker가 관찰된 결과, robot-reachable zone 또는 final-paper configuration이 아니다.
- **Boundary policy:** Boundary point는 `BOUNDARY`와 모든 touching zone ID를 반환한다. 공유 edge는 허용하지만 polygon interior overlap은 금지하며 한 zone을 silently 선택하지 않는다.
- **Goal center:** Zone center는 nominal destination point일 뿐 robot footprint, object footprint, gripper offset, orientation, collision, clearance가 검증된 robot goal pose가 아니다.
- **Visualization:** World→undistorted inverse Homography는 offline visualization에만 사용하며 D011의 raw→world geometry를 변경하지 않는다.
- **Final-paper replacement:** Final camera/workspace, physical destination marker와 robot/object dimensions가 확정된 뒤 별도 final zone config를 생성한다.
- **Status:** ACCEPTED FOR OFFLINE DEVELOPMENT (2026-09-17).

## D017 — Language Interpretation and Physical Grounding Separation

- **Decision:** LLM은 natural-language instruction에서 `action`, `target_phrase`, `destination_phrase`만 추출한다. Pixel/bbox/world coordinate, zone ID/polygon, robot/grasp pose, feasibility, path, velocity와 motor command를 생성하지 않는다.
- **Validation:** Model JSON은 Python schema/action/missing/ambiguity validator를 통과해야 한다. Missing field, unknown zone, referential target와 unsupported action을 임의 보정하거나 실행하지 않는다.
- **Target responsibility:** Approved target phrase만 Task-Conditioned VLM Grounding에 전달하고, VLM result와 deterministic geometry가 target world coordinate를 제공한다.
- **Destination responsibility:** Destination phrase는 D016의 resolver로 확인하며 metric polygon/center는 `zones.json`에서만 가져온다. LLM이 출력한 zone ID나 meter coordinate는 신뢰하지 않는다.
- **Execution separation:** Language `VALID_TASK`와 target `active_workspace_status`를 별도 보존한다. Parser 성공은 physical feasibility나 execution 가능성을 의미하지 않는다.
- **Evidence:** English valid 3개는 end-to-end offline integration됐지만 한국어 target canonicalization 3개와 unsupported-action 1개가 실패했다.
- **Status:** ACCEPTED WITH REWORK REQUIRED (2026-09-17).

## D018 — Surface Extraction and Deterministic Task Canonicalization

- **Decision:** LLM은 `action_phrase`, `target_phrase`, `destination_phrase`의 surface language만 추출한다. Canonical symbolic action, reviewed target grounding phrase와 zone ID는 deterministic Python이 결정한다.
- **Action safety:** Raw instruction의 explicit unsupported expression을 model output보다 먼저 검사한다. `회전시켜`/`rotate` 같은 guard match는 destination missing이나 hallucinated supported action보다 우선하며 grounding/zone/path를 실행하지 않는다.
- **Target policy:** `/home/binci/vlm_ws/config/task_language.json`의 human-reviewed development alias만 known target으로 normalize한다. 명확한 unknown object는 `OPEN_VOCABULARY`로 보존하고 referential phrase는 `TARGET_AMBIGUOUS`로 거부한다.
- **Destination policy:** D016의 zone resolver와 `zones.json` geometry는 변경하지 않는다. LLM이 zone ID나 metric geometry를 생성하지 않는다.
- **Execution separation:** `VALID_TASK`와 target workspace/feasibility 상태는 별도다. White bag language task는 valid이지만 target은 계속 `OUTSIDE`이며 feasibility는 `NOT_EVALUATED`다.
- **Evidence:** 기존 11-case core set이 valid 6/6와 invalid 5/5, 추가 English rotate가 1/1을 기록했다.
- **Status:** ACCEPTED FOR OFFLINE DEVELOPMENT (2026-09-17).

## D019 — Unified Workspace / Task State as a Representation Layer

- **Decision:** Unified Workspace / Task State는 validated language, task-conditioned target grounding, deterministic zone, semantic scene entities와 workspace bounds를 결합하는 representation layer다. Physical feasibility를 판단하거나 motion을 생성하지 않는다.
- **Authority:** 요청 target에는 STEP 08 task-conditioned grounding이 authoritative하다. STEP 07 scene association은 bbox IoU 기반 `SAME_VISUAL_ENTITY_CANDIDATE`이며 optional이다. Association 실패는 valid task representation을 무효화하지 않는다.
- **Validation:** Task는 `VALID_TASK`, target은 `TARGET_FOUND`, destination zone은 configured ID여야 한다. Metric frame/unit 일치, finite target/scene world coordinates, valid destination polygon과 ordered workspace bounds를 확인하며 silent conversion하지 않는다.
- **Deferred:** Zone center는 robot goal pose가 아니다. Robot/object/gripper footprint, obstacle role, feasibility, grasp/approach pose와 path는 생성하지 않는다.
- **Evidence:** Saved inputs만으로 red cup, white bag, trophy state 3/3을 생성했다. White bag `OUTSIDE`도 `VALID_REPRESENTATION`으로 보존했고 system/`qwen_vl` test가 60/60 PASS했다.
- **Status:** ACCEPTED FOR OFFLINE DEVELOPMENT (2026-09-17).

## D020 — Physical Geometry Evidence and Planning-Role Separation

- **Decision:** Planner footprint는 confirmed physical/reference data 또는 명시적인 `DEVELOPMENT_SYNTHETIC` geometry에서만 만든다. VLM image bbox와 bottom-center point는 physical size/polygon source가 아니다.
- **Robot source:** Existing ROS2 Nav2 `base_link` footprint `±0.145 m × ±0.123 m`를 recorded planning geometry로 재사용한다. Source에 함께 있던 padding/inflation 값은 물리 치수가 아니므로 STEP 13에서 제외한다.
- **Unknown policy:** Gripper, task object와 selected obstacle의 확인된 치수가 없으면 null과 `UNKNOWN`을 유지한다. `INFERRED` 값은 audit에 기록할 수 있지만 `READY`나 planner footprint로 승격하지 않는다.
- **Role policy:** STEP 07 semantic type과 planning role을 분리한다. Config가 `TASK_TARGET`, `DYNAMIC_ENTITY`, `STATIC_OBSTACLE_CANDIDATE`, `UNASSIGNED`를 결정하며 VLM이 collision role을 직접 선택하지 않는다.
- **Execution separation:** Physical layer는 representation과 readiness만 계산한다. Collision, clearance, inflation, robot+object union, grasp/approach pose와 path는 생성하지 않는다.
- **Evidence:** 세 STEP 12 state가 별도 physical state로 생성됐고 robot `READY`, 나머지 핵심 geometry `NOT_READY`, overall `PARTIAL`을 반환했다. System/`qwen_vl` test는 72/72 PASS했다.
- **Status:** ACCEPTED FOR OFFLINE DEVELOPMENT (2026-09-17).

## D021 — Robot-Only Static Collision Geometry Baseline

- **Development priority:** Current implementation prioritizes robot-only path planning and dynamic replanning before manipulation.
- **Manipulation deferral:** Gripper and pick-and-place are deferred until the navigation/replanning stack is stable and final gripper geometry is available.
- **Decision:** STEP 14는 confirmed rectangular robot footprint를 yaw 0으로 고정하고, workspace boundary erosion과 axis-aligned obstacle Minkowski expansion으로 continuous metric robot-center collision geometry를 만든다.
- **Obstacle provenance:** 실측 static obstacle이 없으므로 `/home/binci/vlm_ws/config/static_obstacles.json`의 한 rectangle만 `DEVELOPMENT_SYNTHETIC`으로 사용한다. VLM bbox, scene entity 또는 실제 환경 측정값으로 해석하지 않는다.
- **Boundary policy:** Robot footprint가 raw workspace 또는 raw obstacle에 접촉하면 `COLLISION`이다. Valid-center workspace 경계는 query에 포함하지 않고 expanded obstacle 경계는 collision에 포함한다.
- **Margin policy:** STEP 14 safety margin, footprint padding과 inflation은 모두 0이다. Nav2 tuning 값을 물리 치수나 본 단계 margin으로 재사용하지 않는다.
- **Readiness:** Robot-only static planning geometry는 `READY`다. Gripper/object/실측 obstacle geometry가 없는 overall manipulation geometry는 `PARTIAL`을 유지한다.
- **Deferred:** Occupancy grid, A*, path, dynamic replanning, robot execution과 manipulation geometry는 생성하지 않는다.
- **Evidence:** Synthetic obstacle 내부·외부·정확한 접촉 및 workspace 접촉, frame/unit/non-finite/invalid geometry를 검증했고 system 및 `qwen_vl` Python test가 각각 91/91 PASS했다.
- **Status:** ACCEPTED FOR OFFLINE DEVELOPMENT (2026-09-17).

## D022 — Binary Occupancy, Safety Cost, and Deterministic A* Baseline

- **Research role:** A* is used as a deterministic baseline planner, not as the research novelty.
- **Layer separation:** Hard collision constraints and soft safety costs are represented separately. Occupied cells are blocked and never encoded as free-space risk 1.0.
- **Clearance source:** The safety cost layer operates on clearance from the robot-expanded configuration-space obstacle and valid robot-center workspace boundary.
- **Costmap scope:** The current cost layer is an offline development representation and is not ROS2 Nav2 Costmap2D.
- **Grid policy:** Raw workspace origin과 0.02 m resolution을 사용한 cell-center grid다. Metric point는 half-open containing cell에 mapping하며 invalid start/goal을 nearest free cell로 옮기지 않는다.
- **Movement policy:** Fixed order `N, NE, E, SE, S, SW, W, NW`의 8-connected motion을 사용한다. Diagonal step은 두 adjacent orthogonal cell이 모두 free일 때만 허용한다.
- **Objective:** Distance-only는 metric movement distance만 사용한다. Safety-cost는 `distance + safety_weight × destination_risk × distance`를 사용하고 Euclidean heuristic에는 risk penalty를 넣지 않는다.
- **Metric policy:** Accumulated safety risk는 `Σ(destination risk × edge length)`다. Planning time은 monotonic timer로 측정하지만 semantic determinism 대상에서 제외한다.
- **Deferred:** Smoothing, parameter optimization, dynamic replanning, ArUco/real robot integration과 manipulation은 구현하지 않는다.
- **Evidence:** 동일 STEP 14 world에서 두 collision-free path를 생성했고 semantic reconstruction 및 system/`qwen_vl` 122/122 tests가 PASS했다.
- **Status:** ACCEPTED FOR OFFLINE DEVELOPMENT (2026-09-17).

## D023 — Static Planner Robustness Before Dynamic Replanning

- **Decision:** Static planner robustness must be demonstrated across multiple synthetic geometries before dynamic replanning is introduced.
- **Parameter policy:** Planner parameters remain development parameters; STEP 16 characterizes sensitivity but does not optimize or certify them.
- **Scenario policy:** S1–S6 obstacle geometry, endpoints and expected high-level result만 config에 둔다. Exact path length, clearance, risk 또는 planning time을 expected 값으로 hard-code하지 않는다.
- **Execution policy:** 모든 valid scenario에서 distance-only와 safety-cost planner가 동일 collision world, grid, start와 goal을 사용한다. `NO_PATH_EXPECTED`는 nearest goal, clipping 또는 partial success로 보정하지 않는다.
- **Validation policy:** PATH_FOUND는 waypoint occupancy와 continuous collision query, valid-center bounds, 8-connected adjacency 및 no-corner-cutting을 planner 결과와 별도로 재검사한다.
- **Sensitivity policy:** Resolution, safety radius와 safety weight는 one-factor-at-a-time으로 평가한다. Safety radius/weight는 hard occupancy 또는 robot-expanded collision bounds를 변경하지 않는다.
- **Interpretation:** Clearance는 geometric development metric일 뿐 camera/control/ArUco/wheel-slip/physical-geometry uncertainty를 포함한 safety certificate가 아니다. Observed trends와 general conclusions를 구분한다.
- **Evidence:** 여섯 scenario의 12/12 planner status가 expectation과 일치했고, PATH_FOUND 10/10은 collision-free, blocked 2/2는 `NO_PATH`였다. 11개 parameter setting을 완료했고 153/153 tests가 양 Python 환경에서 PASS했다.
- **Deferred:** Parameter optimization, dynamic obstacle update, path invalidation/replanning, real robot 및 manipulation은 실행하지 않는다.
- **Status:** ACCEPTED FOR OFFLINE DEVELOPMENT (2026-09-18).

## D024 — Dynamic State Update and Path Invalidation Before Replanning

- **Separation:** Dynamic obstacle state updates are separated from automatic replanning. 각 scenario의 초기 `SAFETY_COST_ASTAR` path는 한 번만 생성하고 event 이후에는 수정하거나 A*를 다시 실행하지 않는다.
- **State model:** Stable obstacle ID와 순차 정수 `world_revision`을 사용한다. `ADD` existing ID, `REMOVE`/`MOVE` unknown ID와 비연속 event index는 명시적으로 거부하며 timestamp를 identity로 사용하지 않는다.
- **World rebuild:** 각 event는 raw metric obstacle state에서 기존 STEP 14 collision builder를 호출하고 STEP 15 occupancy/safety layer 전체를 다시 만든다. Expanded JSON이나 grid cell을 직접 patch하지 않으며 additional inflation은 0이다.
- **Invalidation:** Current path invalidation uses hard collision constraints; soft safety-cost changes alone do not invalidate a path. Waypoint/workspace, 8-connected adjacency, continuous metric edge intersection과 diagonal corner-cut을 검사한다.
- **Provenance:** `BLOCKED`는 최초 waypoint/edge index, reason, blocking obstacle ID와 `replan_required=true`를 기록한다. 점유 셀을 포함한 blocked path의 accumulated risk는 `NOT_APPLICABLE_PATH_BLOCKED`다.
- **Development scope:** STEP 17 uses synthetic offline obstacle events and is not live-camera dynamic perception. Timeout, persistence, tracking, automatic replanning, robot execution과 manipulation은 구현하지 않는다.
- **Evidence:** D1–D6의 8/8 event가 기대 상태와 일치했고 system/`qwen_vl` Python에서 189/189 tests가 PASS했다.
- **Status:** ACCEPTED FOR OFFLINE DEVELOPMENT (2026-09-18).

## D025 — Current-Pose Event-Driven Automatic Replanning

- **Start policy:** Dynamic replanning starts from the current robot pose, not the original mission start. Offline progress는 active path의 exact waypoint index로 표현하며 nearest-free snapping이나 fractional interpolation을 사용하지 않는다.
- **Trigger policy:** Only hard path invalidation triggers automatic replanning in the current baseline. Soft clearance/risk 변화만 있는 `VALID` path에서는 planner를 실행하지 않는다.
- **Retention policy:** A valid active path is retained even if environmental changes later permit a potentially better route. Blocker removal만으로 superseded path를 복원하거나 cost-improvement replan을 실행하지 않는다.
- **Planner policy:** Updated world에서 rebuild한 occupancy/safety layer와 기존 `SAFETY_COST_ASTAR`만 사용한다. 새 A*, smoothing, substitute goal과 partial path activation은 금지한다.
- **Switch policy:** New path는 waypoint/world collision, workspace, adjacency, corner-cut, continuous edge, current-pose start mapping과 original-goal mapping 검증을 모두 통과한 후에만 `ACTIVE`가 된다. 이전 path는 `SUPERSEDED`로 보존한다.
- **Failure policy:** Updated robot pose collision은 `ROBOT_POSE_INVALID`, goal collision은 `GOAL_BLOCKED`로 planner를 실행하지 않고 stop한다. A* `NO_PATH`는 `REPLAN_FAILED / STOP_REQUIRED`이며 partial path를 활성화하지 않는다.
- **Development scope:** STEP 18 is an offline event-driven replanning baseline, not live real-time navigation. Persistence, debounce, live perception, ROS/Nav2 execution과 manipulation은 포함하지 않는다.
- **Evidence:** R1–R7의 9/9 event가 기대 state/action/revision과 일치했고 system/`qwen_vl` Python에서 229/229 tests가 PASS했다.
- **Status:** ACCEPTED FOR OFFLINE DEVELOPMENT (2026-09-18).

## D026 — Logical-Time Dynamic Obstacle Persistence Policy

- **Decision:** Hard path blockage no longer triggers immediate replanning. The development policy first stops and observes obstacle persistence.
- **Temporary obstacle policy:** Temporary blocking obstacles clear and resume the existing active path without replanning, but only after absence debounce and full-world path revalidation return `VALID`.
- **Persistent obstacle policy:** Persistent blocking obstacles trigger the existing STEP 18 automatic replanning pipeline exactly once per blocking episode. A successful plan starts at the stopped active-path waypoint and retains the original mission goal; `NO_PATH` remains `STOP_REQUIRED`.
- **Observation separation:** Raw detections feed presence/absence debounce. Only confirmed obstacles enter the collision world, and persistence timing starts only when that confirmed world actually hard-blocks the remaining active path.
- **Time policy:** STEP 19 uses deterministic logical time and synthetic observations, not live camera timing or wall-clock classification. Equality within the configured tolerance satisfies `blocking_duration >= persistence_threshold`.
- **Parameter limitation:** Persistence and debounce parameters are development parameters and are not physical safety guarantees. `STOP` is an offline navigation-policy decision and not a hardware safety stop.
- **Status:** ACCEPTED FOR OFFLINE DEVELOPMENT (2026-09-18).

## D027 — Fixed-Camera Pixel-Space Scene Monitoring Boundary

- **Loop separation:** The fast camera loop is separated from the slower semantic/VLM loop. STEP 20 performs CPU OpenCV frame differencing and emits a trigger; it does not load or run Qwen.
- **Observation boundary:** Pixel-space scene-change candidates are observations, not planner obstacles. They remain `CHANGE_CANDIDATE / NOT_EVALUATED / NOT_AVAILABLE / NOT_ASSIGNED` and are not injected into STEP 17 or STEP 19.
- **Geometry boundary:** No metric obstacle geometry is inferred directly from image bbox dimensions. STEP 20 does not apply the legacy Homography to a live stream without resolution/pose compatibility validation.
- **Camera selection:** Live capture uses the exact configured external `Ucamera001` by-id path only. A built-in camera is never selected as fallback; absence is recorded as `NOT_AVAILABLE` without changing system configuration.
- **Reference/change policy:** The reference is fixed only after bounded warmup and consecutive stability checks. Local components are area-filtered and tracked deterministically; excessive changed-pixel ratio is `GLOBAL_CHANGE` and does not produce an ordinary candidate or semantic trigger.
- **Identity limitation:** Tracking is pixel-space change continuity, not object identity. Robot motion itself may appear as scene change until robot localization/masking is integrated; merge/split may change identity.
- **Status:** ACCEPTED FOR SOFTWARE/OFFLINE DEVELOPMENT; LIVE EXTERNAL CAMERA NOT AVAILABLE (2026-09-18).

## D028 — Stored-Image Geometry Compatibility and Replay Boundary

- **Installation:** Permanent camera installation is not required for current offline development.
- **Reuse gate:** Stored images may share a legacy workspace transform only when camera-geometry compatibility is sufficiently supported.
- **Resolution:** Same resolution alone is not evidence of the same camera pose.
- **Registration evidence:** Image-to-image feature registration is development compatibility evidence, not final physical camera-pose validation.
- **Replay evidence:** Offline stored-image replay is not evidence of live real-time operation.
- **Classification:** `photo_04.png`만 `REFERENCE_VERIFIED / REFERENCE_ALLOWED`다. 충분한 고정-parameter registration evidence가 있는 raw-like candidate만 `DEVELOPMENT_COMPATIBLE_CANDIDATE / DEVELOPMENT_INFERRED_ALLOWED`이며, provenance를 명시한다.
- **Domain:** Legacy metric pipeline은 raw → stored undistortion → undistorted pixel → Homography 순서를 유지한다. Diagnostic image-to-image transform은 metric Homography가 아니며 VLM, metric grounding 또는 planner 입력으로 재사용하지 않는다.
- **Replay:** Derived/mismatched image는 제외하고 filename order를 deterministic manifest order로만 사용한다. 실제 capture timeline 또는 live tracking evidence로 해석하지 않는다.
- **Status:** ACCEPTED FOR OFFLINE DEVELOPMENT; DATASET `OFFLINE_REPLAY_READY` (2026-09-18).

## D029 — Triggered Semantic Metric Point Boundary

- **Scope:** STEP 22 connects triggered pixel observations to VLM semantics and metric point grounding only.
- **Trigger:** Qwen runs only for `semantic_trigger=true` frames with confirmed STEP 20 candidates from the STEP 21 compatible replay manifest.
- **Association:** Candidate IDs and raw-pixel candidate boxes must be preserved. Only a validated full-image VLM bbox with `ASSOCIATED` status may enter metric grounding; weak or rejected results do not create a point.
- **Geometry:** Metric grounding uses the bottom-center of a validated VLM visible-object bbox through the existing deterministic legacy geometry pipeline. Development-compatible image provenance is retained.
- **Safety boundary:** Scene-change candidate geometry is never used directly as physical obstacle geometry. A metric point is not a physical obstacle footprint. Planner injection remains prohibited until obstacle footprint geometry has an explicit source.
- **Representative policy:** Exhaustive inference is stopped. Each new run receives the original full-resolution source image and exactly one confirmed candidate; at most three representative attempts are allowed.
- **Result:** Eight preserved exhaustive/retry runs and three representative CPU runs produced no valid associated metric point. Representative attempts all parsed as `NO_VISIBLE_ENTITY`; malformed exhaustive bbox domains were rejected rather than repaired or threshold-tuned. Software/inference/strict gates PASS, representative integration PARTIAL.
- **Status:** ACCEPTED AS PARTIAL EVIDENCE (2026-09-18).

## D030 — Candidate-Conditioning Coordinate Unification

- **Confirmed bbox convention:** STEP 20 connected-component boxes are half-open `[x1,y1,x2,y2)` because OpenCV `CC_STAT_WIDTH/HEIGHT` are stored as `x+width/y+height`. An endpoint equal to image width or height is valid.
- **Prompt audit:** STEP 22 distinguished raw candidate metadata from normalized object output, but omitted image dimensions, explicit min/max order and half-open boundary semantics. This is classified `LIKELY_CONFUSING`, not a proven causal explanation.
- **Coordinate prototype:** Audit-only conversion uses `floor(raw/extent*1000+0.5)` and is marked `AUDIT_DERIVED_COORDINATE`; it never repairs a prior model output.
- **Recommended representation:** STEP 22C should evaluate Method C: same-resolution full-scene visible overlay plus normalized integer candidate coordinates and exact track ID. No crop/overlay coordinate may enter metric geometry.
- **Separation:** Scene-change candidate geometry remains separate from full-object semantic geometry. Validated original full-image object bbox bottom-center remains the only semantic-to-metric input.
- **Status:** ACCEPTED FOR STEP 22C DESIGN; NO MODEL INFERENCE IN STEP 22B (2026-09-18).

## D031 — Method C Single-Inference Boundary

- **Selection:** STEP 22C는 `photo_08 / track_039` 하나와 STEP 22B의 2016×1133 highlight overlay만 사용한다. Candidate metadata는 normalized integer `[504,481,682,604]`만 prompt에 제공한다.
- **Semantic-only overlay:** STEP 22C uses candidate overlay and normalized candidate metadata solely for semantic attention. Overlay pixels are never used as Homography input.
- **Metric source:** Metric coordinates are derived only from a validated full-object bbox mapped back to the original full-image domain. Raw candidate bbox는 provenance/association 전용이다.
- **Execution bound:** One representative actual inference was executed. Retry, fallback candidate, Method D와 planner는 실행하지 않았다.
- **Result:** Strict parser가 `NO_VISIBLE_ENTITY`를 수용했으므로 association `NOT_EVALUATED`, metric `NOT_RUN`, STEP 22C/STEP 22는 `PARTIAL`이다. 이 단일 결과로 원인을 확정하지 않는다.
- **Status:** ACCEPTED AS PARTIAL EVIDENCE (2026-09-18).

## D032 — Method D Two-Image Semantic Conditioning Boundary

- **Inputs:** IMAGE 1은 original 2016×1133 `photo_08`, IMAGE 2는 동일 image의 STEP 22B 15% context crop이다. Method D uses a secondary crop only to improve semantic visibility.
- **Output reference:** Visible bbox는 explicit `bbox_reference_image=IMAGE_1`과 integer normalized 0–1000 domain을 만족해야 한다. Crop-relative output은 remap하지 않는다.
- **Metric source:** All metric coordinates continue to originate from a full-object bbox expressed in the original full-image coordinate system. Crop coordinates are never mapped through Homography.
- **Execution bound:** STEP 22D executes exactly one actual Qwen inference. 최초 visualization 후처리 오류는 preserved raw output으로만 재개했으며 generation을 반복하지 않았다.
- **Result:** Tennis-racket full-image bbox가 `ASSOCIATED`됐고 existing bottom-center→stored undistortion→legacy Homography가 `[0.764672,0.713486] m`를 생성했다. 이는 development point이며 physical footprint가 아니다.
- **Status:** ACCEPTED; STEP 22 SEMANTIC-METRIC INTEGRATION PASS (2026-09-19).

## D033 — Physical Obstacle Geometry Source and Anchor Gate

- **Boundary:** A semantic metric point is not itself a physical obstacle footprint. Physical object geometry requires an explicit dimensional source.
- **Prohibited inference:** The VLM bounding box is not used to infer metric width or length. Semantic labels, VLM-generated sizes and unvalidated bbox corners are not physical geometry sources.
- **Anchor:** Observed bottom-center is not automatically the physical footprint center. Unknown anchor relation blocks world footprint placement even if dimensions later become known.
- **Layers:** Physical geometry, uncertainty, safety margin, and planner representation are separate layers. STEP 23 generates no planning geometry, safety inflation, collision world or path.
- **Representative result:** Repository audit found no explicit tennis-racket physical measurement or anchor evidence, so the STEP 22D point remains available while physical geometry is `NOT_READY_MEASUREMENT_REQUIRED`.
- **Verification:** STEP 14 `dev_obstacle_01` is reused only as `STEP14_EXISTING_DEVELOPMENT_SYNTHETIC` to test local-to-world translation; it is not linked to the observed racket.
- **Status:** ACCEPTED; STEP 23 SOFTWARE PASS / REPRESENTATIVE MEASUREMENT REQUIRED (2026-09-19).

## D034 — User Measurement Intake and READY-Only Registry

- **Input authority:** Representative geometry may enter the real registry only from `USER_PHYSICAL_MEASUREMENT`; internet/default dimensions, semantic assumptions, VLM results, image bboxes and synthetic fixtures are rejected as actual evidence.
- **Missing-input behavior:** When `config/representative_object_measurement.json` is absent, STEP 23M creates no actual dimensions and returns `WAITING_FOR_USER_MEASUREMENT` with a null-only template and measurement request.
- **Canonicalization:** Raw values in `mm`, `cm` or `m` are preserved and deterministically converted to meters. Raw input and canonical metric values remain separate provenance fields.
- **Readiness:** Dimensions alone are insufficient. Measured/validated anchor and required orientation are mandatory; incomplete records cannot enter `measured_object_geometry_registry.json`.
- **Boundary:** READY physical geometry still has `planning_geometry=NOT_GENERATED`; STEP 17/18/19, A*, costmap and planner injection remain prohibited.
- **Status:** ACCEPTED; STEP 23M SOFTWARE PASS / WAITING FOR USER MEASUREMENT (2026-09-19).

## D035 — Semantic-Metric Point to Navigation Approximation Boundary

- **Layer decision:** Physical geometry and navigation obstacle approximation are separate layers. Missing physical measurements do not block generation of a validated approximate navigation observation.
- **Reference decision:** STEP 24N uses only the preserved STEP 22D semantic-metric reference point. `BBOX_BOTTOM_CENTER_DEVELOPMENT_GROUND_REFERENCE` is not a physical/object/footprint center.
- **Representation decision:** The continuous navigation point is deterministically projected with the existing STEP 15 helper and configured resolution to one containing grid cell. The cell is navigation discretization, not inferred object size, and does not mutate occupancy in STEP 24N.
- **Inference decision:** Object dimensions, radius or shape are never inferred from the VLM bbox or semantic label. No new safety margin is added; clearance remains the existing collision/cost layer's responsibility.
- **Integration boundary:** The current STEP 17 contract accepts development-synthetic axis-aligned rectangles, not point observations. A separately approved STEP 25N bridge is required before dynamic-world insertion.
- **Physical branch:** STEP 23M remains available for later manipulation or precise physical interaction and remains `WAITING_FOR_USER_MEASUREMENT`.
- **Status:** ACCEPTED FOR OFFLINE NAVIGATION ADAPTER DEVELOPMENT (2026-09-19).

## D036 — Navigation Cell to STEP 17 Dynamic-World Bridge

- **Bridge:** A STEP 24N `SINGLE_OCCUPIED_CELL` becomes the exact metric AABB of that grid cell for STEP 17 compatibility. The AABB is `NAVIGATION_DISCRETIZATION_CELL`, not physical footprint or measured object geometry.
- **Provenance:** Observed records use `NAVIGATION_APPROXIMATION / STEP24N_NAVIGATION_APPROXIMATION`; they are never relabeled `DEVELOPMENT_SYNTHETIC`. Existing development-synthetic scenarios remain accepted unchanged.
- **Traceability:** The STEP 22D reference point, STEP 24N row/column and cell center, STEP 25N cell bounds, source image and track are retained together.
- **Identity/time:** Dynamic ID is deterministic and observation-scoped. `track_039` is not permanent object identity; timestamp remains `NOT_AVAILABLE`, while event index means deterministic offline order only.
- **Occupancy interpretation:** The representative source cell was already occupied and remains 1→1. Existing robot-footprint collision expansion causes 78 additional binary cells to become occupied, so source-cell delta alone cannot prove or disprove dynamic-world registration.
- **Boundary:** No new radius, safety margin or object inflation is introduced. STEP 17 state/collision/grid materialization is allowed; path validation, STEP 18/19, A* and replanning remain deferred.
- **Status:** ACCEPTED; STEP 25N PASS (2026-09-19).

## D037 — Same-Path Before/After Validity Attribution

- **Path authority:** STEP 26N reuses the preserved STEP 17 D1 current path because it is the canonical artifact aligned with the STEP 25N D1 before state. It never generates a replacement path.
- **Comparison rule:** The exact same grid/world waypoint sequence and digest must be evaluated against world revisions 0 and 1.
- **Causality rule:** `AFTER=BLOCKED` alone is insufficient. Observation-induced blocking requires `BEFORE=VALID`, `AFTER=BLOCKED` and supporting changed-cell or dynamic-obstacle provenance.
- **Representative result:** The path is `VALID → VALID`, therefore `NO_PATH_EFFECT`; blocking location and dynamic attribution are not claimed.
- **Occupancy interpretation:** The source cell was already occupied and STEP 25N added 78 robot-collision occupancy cells, but none invalidated this established path.
- **Boundary:** STEP 26N evaluates validity only. STEP 18/19, A*, replanning and alternate-path generation remain prohibited.
- **Status:** ACCEPTED; STEP 26N PASS / REPRESENTATIVE NO PATH EFFECT (2026-09-19).

## D038 — Controlled Path-Effect Evidence Separation

- **Control authority:** CASE A is the preserved STEP 26N observed navigation approximation. Its `VALID → VALID / NO_PATH_EFFECT` result is not altered or relabeled.
- **Positive-fixture authority:** CASE B reuses documented STEP 17 D2 `D2_add_on_path`, its stored current path and configured `dynamic_box_02`; STEP 27N invents no path, obstacle or coordinate.
- **Comparison rule:** Both cases use the existing validity contract. A positive effect requires the same path before/after, `VALID → BLOCKED`, and recorded blocking waypoint/edge/cell/obstacle provenance.
- **Evidence boundary:** CASE A remains `OBSERVED_NAVIGATION_APPROXIMATION`; CASE B remains `DOCUMENTED_DEVELOPMENT_FIXTURE`. Development fixture behavior is software-contract evidence, not camera/VLM evidence.
- **Execution boundary:** STEP 18/19, A*, new-path generation, replanning, Qwen, Homography and physical-geometry inference are prohibited.
- **Status:** ACCEPTED; STEP 27N PASS / CONTROLLED VALID AND BLOCKED CONTRACT VERIFIED (2026-09-19).

## D039 — Validated Path Block to Decision-Only Persistence Policy

- **Status separation:** `PATH_BLOCKED != PERSISTENT_BLOCK_CONFIRMED != REPLANNING_EXECUTED`. STEP 27N supplies validity evidence; the existing STEP 19 state machine alone decides temporal persistence.
- **Control rule:** CASE A `VALID` remains `MOVING / CONTINUE_CURRENT_PATH`; dynamic-obstacle presence alone cannot trigger STOP/WAIT.
- **Development evidence:** CASE B remains `DOCUMENTED_DEVELOPMENT_FIXTURE`. Its mapped STEP 19 P2 logical replay is software-policy evidence, not repeated live-camera evidence.
- **Parameter authority:** Existing STEP 19 config remains authoritative: presence 2, absence 2, 0.2 s interval and 1.0 s persistence threshold. No threshold or state-machine redesign is permitted.
- **Decision boundary:** `execute_replanning=False` preserves the existing default behavior while allowing `PERSISTENT_BLOCKING / REPLAN_REQUIRED` as a decision-only output. Planner runs remain 0, path revision remains 0, and STEP 18/A*/new-path generation remain `NOT_RUN`.
- **Safety semantics:** `STOPPED_FOR_OBSTACLE`, `WAITING_FOR_CLEARANCE` and `STOP_AND_WAIT` are offline policy states/actions, not verified motor or hardware-stop behavior.
- **Physical branch:** STEP 23M remains `WAITING_FOR_USER_MEASUREMENT`; physical geometry remains `NOT_READY` and unused.
- **Status:** ACCEPTED; STEP 28N PASS / PERSISTENT BLOCK DECISION READY, REPLANNING NOT EXECUTED (2026-09-19).

## D040 — Persistent Decision to Existing STEP 18 Replanning Gate

- **Gate:** STEP 18 is eligible only for the conjunction `BLOCKED + PATH_BLOCKED + PERSISTENT_BLOCKING + REPLAN_REQUIRED` with valid documented blocking and STEP 19 decision provenance. `VALID`, `NO_PATH_EFFECT`, `TEMPORARY_BLOCKING`, missing decisions and malformed provenance are rejected.
- **Control:** CASE A remains `OBSERVED_NAVIGATION_APPROXIMATION / VALID / CONTINUE_CURRENT_PATH`; it invokes STEP 18 zero times and keeps revision 0.
- **Positive evidence:** CASE B remains `DOCUMENTED_DEVELOPMENT_FIXTURE`, not camera/VLM-derived evidence. It preserves STEP 17 D2 `dynamic_box_02`, world revision 1, stored path, waypoint 20 from the STEP 28N trace, original goal and unchanged planner config.
- **Implementation:** STEP 29N calls existing STEP 18 `evaluate_event_and_replan` once with `apply_obstacle_update=False`; it adds no planner, cost, heuristic, geometry or revision logic.
- **Actual result:** Existing STEP 18 returned `REPLAN_SUCCESS / SWITCH_TO_REPLANNED_PATH`; old revision 0 became `SUPERSEDED`, and the accepted `VALID` replacement became revision 1. This verifies offline software integration only.
- **Failure semantics:** Existing no-path behavior remains `REPLAN_FAILED / STOP` with no fabricated replacement, as preserved by the historical STEP 18 R4 contract artifact.
- **Boundaries:** STEP 23M remains `WAITING_FOR_USER_MEASUREMENT`; Qwen, Homography, ROS/Nav2 and hardware execution remain `NOT_RUN`.
- **Status:** ACCEPTED; STEP 29N PASS / EXISTING STEP 18 OFFLINE HANDOFF VERIFIED (2026-09-20).

## D041 — Dual-Track Offline Navigation Evidence Boundary

- **System structure:** STEP 30N preserves two independent tracks instead of manufacturing one continuous observed-positive story.
- **Observed control:** TRACK A is `OBSERVED_NAVIGATION_APPROXIMATION` and camera/VLM-derived. STEP22D→24N→25N→26N identity is preserved, but its result is `VALID / NO_PATH_EFFECT / CONTINUE_CURRENT_PATH`; STEP18 calls remain zero.
- **Development positive:** TRACK B is `DOCUMENTED_DEVELOPMENT_FIXTURE` and not camera/VLM-derived. STEP17 D2→27N→28N→29N preserves `dynamic_box_02`, stored path identity, persistent decision, one existing STEP18 invocation, `REPLAN_SUCCESS` and valid revision 1 replacement.
- **Allowed claim:** An actual semantic-metric observation can enter approximate navigation without measured physical dimensions and correctly leave an unaffected path unchanged; a separate development fixture verifies downstream blocking, persistence and replanning.
- **Prohibited claim:** The observed tennis racket caused replanning. No observed `VALID → BLOCKED` example currently exists.
- **Physical/safety boundary:** Navigation approximation does not supply physical footprint accuracy, certified clearance, hardware safety or manipulation readiness. STEP23M remains `WAITING_FOR_USER_MEASUREMENT`.
- **Execution boundary:** STEP30N is artifact-only verification; Qwen, Homography, earlier STEP runners, ROS/Nav2 and hardware execution are not run.
- **Status:** ACCEPTED; STEP 30N PASS / DUAL-TRACK OFFLINE SYSTEM LINEAGE VERIFIED (2026-09-20).
