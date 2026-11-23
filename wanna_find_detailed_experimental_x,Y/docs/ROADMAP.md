# Roadmap

각 Phase는 선행 dependency가 `PASS`인 뒤 구현한다. 아래 output 경로는 계획이며 실제 결과가 생기기 전에는 생성하지 않는다.

## Phase 0 — Experimental Environment Freeze

- **Purpose:** camera, workspace, robot, marker, object, gripper, goal, 속도와 safety margin을 고정한다.
- **Existing Asset:** calibration 해상도 1920×1080; Mecanum footprint/kinematic 값 일부; ArUco 187/188 mm 후보.
- **Missing Work:** 실제 camera model/FPS/pose, 좌표계, 모든 물리 치수와 marker-to-robot transform 확정.
- **Implementation:** 실측값과 측정 방법을 `config/experiment.yaml`에 단일 출처로 기록한다.
- **Experiment:** config 값과 실제 hardware를 항목별 대조하고 사진/측정 기록을 남긴다.
- **Metrics:** completeness, 단위/좌표계 일관성, 반복 측정 편차.
- **PASS Gate:** 모든 필수 항목이 측정 근거와 함께 고정되고 상충 값이 없다.
- **Output Files:** `config/experiment.yaml`, config 세부 파일, freeze 기록.
- **Dependencies:** 없음.

## Phase 1 — Independent Geometry Validation

- **Purpose:** undistorted 12-point homography를 생성에 쓰지 않은 바닥점으로 검증한다.
- **Existing Asset:** calibration JSON, `homography_12pt_undistorted.json`, 적합/LOO 검사 코드.
- **Missing Work:** 현재 설치 확인, 독립 20~30점 dataset, region 정의, 재현 가능한 metric/figure 생성.
- **Implementation:** raw pixel을 동일 calibration으로 undistort한 뒤 저장된 H로 world X/Y를 계산하고 CSV와 summary를 만든다.
- **Experiment:** 중앙·가장자리의 독립 점에서 raw `(u,v)`와 실측 `(X,Y)`를 수집한다. 기존 12점은 제외한다.
- **Metrics:** mean, median, RMSE, max, center RMSE, edge RMSE.
- **PASS Gate:** Phase 0에서 사전 확정한 tolerance를 만족하고 모든 metric이 raw data에서 재계산된다.
- **Output Files:** `datasets/raw/EXP001_*`, `datasets/annotations/EXP001_points.csv`, `results/EXP001_runNN/`.
- **Dependencies:** Phase 0의 camera/workspace freeze.

## Phase 2 — Robot World Localization

- **Purpose:** top ArUco로 workspace 기준 robot `(x,y,yaw)`를 출력한다.
- **Existing Asset:** marker 검출, corners, camera-relative pose, trajectory CSV.
- **Missing Work:** workspace transform, yaw convention, marker-center offset, stale/validity, 정량 ground truth.
- **Implementation:** 검출과 deterministic transform을 분리하고 robot reference pose와 timestamp를 출력한다.
- **Experiment:** 알려진 여러 X/Y 및 yaw에서 반복 측정한다.
- **Metrics:** XY RMSE, yaw MAE, detection rate, latency.
- **PASS Gate:** 사전 tolerance와 detection/latency 기준을 충족하고 offset이 검증된다.
- **Output Files:** localization config/code/test, `results/EXP002_runNN/`.
- **Dependencies:** Phase 0, Phase 1 PASS.

## Phase 3 — External Vision Robot Pose Control

- **Purpose:** 외부 camera pose feedback으로 goal `(x,y,yaw)`에 이동한다.
- **Existing Asset:** ROS2 Mecanum controller, wheel odometry, gyro-z EKF, command timeout.
- **Missing Work:** laptop↔Jetson pose/control interface, error controller, stale/lost fail-stop, dry-run.
- **Implementation:** 검증된 pose와 goal error를 bounded `vx,vy,omega`로 변환하고 모든 failure에서 zero command를 보낸다.
- **Experiment:** forward, backward, strafe, diagonal, rotation, combined motion.
- **Metrics:** trajectory RMSE, final XY/yaw error, settling time, stop latency.
- **PASS Gate:** 각 motion이 tolerance를 충족하고 marker/camera/communication failure 시 정지가 입증된다.
- **Output Files:** control code/config/tests, `results/EXP003_runNN/`.
- **Dependencies:** Phase 2 PASS; 기존 ROS2 baseline 재검증.

## Phase 4 — VLM Object Grounding

- **Purpose:** 자연어 target을 structured bbox로 식별하고 metric coordinate에 연결한다.
- **Existing Asset:** local Qwen3-VL 4B loading, single/multi bbox prompt, bottom-center 변환 예제.
- **Missing Work:** schema validation, 원문 응답 보존, 기준 geometry 통합, dataset과 정량 평가.
- **Implementation:** image+text→validated `{label,bbox}`와 별도 geometry 함수의 world X/Y로 분리한다.
- **Experiment:** 대상/비대상/복수 객체 dataset에서 반복 inference 후 annotation과 비교한다.
- **Metrics:** target accuracy, wrong-target rate, position RMSE, inference latency, parse failure rate.
- **PASS Gate:** 사전 accuracy/error 기준을 만족하고 VLM이 metric 값을 생성하지 않는다.
- **Output Files:** perception/geometry code, annotation dataset, `results/EXP004_runNN/`.
- **Dependencies:** Phase 1 PASS.

## Phase 5 — Zone Grounding

- **Purpose:** 자연어 destination을 predefined zone ID와 연결한다.
- **Existing Asset:** 없음.
- **Missing Work:** zone vocabulary, polygon config, alias와 ambiguity policy.
- **Implementation:** semantic label resolver와 config lookup만 구현한다.
- **Experiment:** zone 표현 dataset을 실제 ID와 비교한다.
- **Metrics:** zone accuracy, unknown/ambiguous rejection rate.
- **PASS Gate:** 모든 평가 지시가 올바른 ID 또는 명시적 rejection으로 처리된다.
- **Output Files:** `config/workspace.yaml`, zone resolver/tests/results.
- **Dependencies:** Phase 0 PASS.

## Phase 6 — Unified Workspace State

- **Purpose:** robot, objects, zones, obstacles, gripper, timestamp/confidence를 한 schema로 통합한다.
- **Existing Asset:** 개념 schema만 정의됨.
- **Missing Work:** types/schema, units/frames, freshness/validation rules.
- **Implementation:** 각 모듈 결과를 복사·검증하여 planner 입력 상태를 생성한다.
- **Experiment:** 정상, 누락, stale, invalid fixture를 offline 검증한다.
- **Metrics:** schema pass/rejection, freshness detection, deterministic serialization.
- **PASS Gate:** invalid/stale state가 실행 경로로 전달되지 않고 valid fixture가 재현 가능하다.
- **Output Files:** world module/schema/tests.
- **Dependencies:** Phase 2, 4, 5 PASS.

## Phase 7 — Natural-Language Task Parsing

- **Purpose:** 지시를 `{action,target,destination,constraints}`로 구조화한다.
- **Existing Asset:** 예시 schema만 정의됨.
- **Missing Work:** parser, schema validation, evaluation dataset.
- **Implementation:** restricted action/schema를 사용하고 미지원/모호한 지시를 거부한다.
- **Experiment:** paraphrase와 invalid instruction annotation dataset을 평가한다.
- **Metrics:** action, target, destination, constraint accuracy와 rejection accuracy.
- **PASS Gate:** 사전 기준을 충족하고 parser가 motor/path/metric 명령을 생성하지 않는다.
- **Output Files:** task module/schema/dataset/results.
- **Dependencies:** Phase 5의 zone vocabulary.

## Phase 8 — Gripper Hardware and Control

- **Purpose:** 전면 gripper를 안전하고 반복 가능하게 open/close한다.
- **Existing Asset:** 없음이 확인됨.
- **Missing Work:** mechanism 선택, 치수/offset, servo interface, feedback/timeout, emergency release 정책.
- **Implementation:** U-guide + single servo/lock 후보 중 최소 구조와 bounded open/close command를 연결한다.
- **Experiment:** 무부하와 대상 물체로 반복 grasp/release한다.
- **Metrics:** grasp success, release success, repeatability, actuation time.
- **PASS Gate:** 목표 반복 횟수에서 안전 기준과 성공률을 충족한다.
- **Output Files:** `config/gripper.yaml`, controller, hardware/test/results 기록.
- **Dependencies:** Phase 0 object/gripper freeze; 사용자 승인 hardware test.

## Phase 9 — Grasp / Approach Pose

- **Purpose:** known object/gripper geometry로 robot approach `(x,y,yaw)`를 계산한다.
- **Existing Asset:** 없음.
- **Missing Work:** object pose convention, grasp point/offset, robot/gripper transforms, visualization.
- **Implementation:** rigid 2D transform으로 단일 configured approach pose를 계산한다.
- **Experiment:** known fixtures에서 수기 기준과 계산/그림을 비교한다.
- **Metrics:** approach XY/yaw numerical error, visualization consistency.
- **PASS Gate:** 기준 fixture 전부 tolerance 내이고 좌표계가 명시된다.
- **Output Files:** object/robot/gripper config, planning code/tests/figures.
- **Dependencies:** Phase 0, 4, 8 PASS.

## Phase 10 — Physical Feasibility Validator

- **Purpose:** 실행 전 boundary, collision, grasp/transport clearance, goal accessibility를 결정론적으로 검사한다.
- **Existing Asset:** Mecanum footprint 후보와 Nav2 footprint config.
- **Missing Work:** workspace/obstacle polygons, object-combined footprint, reason taxonomy와 test cases.
- **Implementation:** geometry library 또는 설치 dependency의 최소 2D collision checks로 `{feasible,reason}`을 반환한다.
- **Experiment:** 경계/충돌/blocked approach/작은 goal 등 synthetic fixtures를 검사한다.
- **Metrics:** expected decision 일치율, false-feasible count.
- **PASS Gate:** 모든 unsafe fixture를 거부하고 false-feasible이 0이다.
- **Output Files:** validator/config/test fixtures/results.
- **Dependencies:** Phase 6, 8, 9 PASS.

## Phase 11 — Path Planning

- **Purpose:** start→approach와 robot+object→goal의 collision-free 2D path를 만든다.
- **Existing Asset:** Nav2 DWB/NavFn config와 robot footprint; `use_astar: false`인 현재 설정.
- **Missing Work:** 본 연구 workspace grid, deterministic A* baseline, 두 footprint, offline visualization.
- **Implementation:** 먼저 최소 occupancy-grid A*를 offline 검증하고 필요할 때 기존 Nav2 실행 계층과 연결한다.
- **Experiment:** free, narrow, blocked, edge, footprint-change maps를 simulation한다.
- **Metrics:** planning success, collision count, path length, planning time.
- **PASS Gate:** feasible fixture는 collision-free path, infeasible fixture는 명시적 failure를 반환한다.
- **Output Files:** planner/config/tests/maps/figures/results.
- **Dependencies:** Phase 10 PASS.

## Phase 12 — Grasp Integration Experiment

- **Purpose:** localization→object pose→approach→plan→move→fine-align→close를 통합한다.
- **Existing Asset:** 선행 Phase 결과 예정.
- **Missing Work:** orchestration, state transition, abort/recovery와 실험 protocol.
- **Implementation:** 단일 물체/고정 시작 조건의 최소 sequential state machine을 사용한다.
- **Experiment:** 반복 approach와 grasp trial.
- **Metrics:** approach success, grasp success, XY/yaw alignment error.
- **PASS Gate:** 사전 trial 수와 success/error 기준을 만족하고 failure cause가 기록된다.
- **Output Files:** integration code/config, `results/EXP012_runNN/`.
- **Dependencies:** Phase 3, 6, 8, 9, 10, 11 PASS.

## Phase 13 — Transport Experiment

- **Purpose:** grasp된 object를 결합 footprint로 goal까지 운반한다.
- **Existing Asset:** Phase 12와 두 번째 path 결과 예정.
- **Missing Work:** holding-state interlock, combined footprint execution, transport protocol.
- **Implementation:** confirmed holding 상태에서만 transport path를 실행하고 failure 시 정지한다.
- **Experiment:** 여러 start/goal에서 반복 운반한다.
- **Metrics:** transport success, collision rate, path length, task time.
- **PASS Gate:** success 기준을 만족하고 collision이 허용 기준(기본 0)을 넘지 않는다.
- **Output Files:** transport integration/results.
- **Dependencies:** Phase 12 PASS.

## Phase 14 — Release and Final Verification

- **Purpose:** release/retreat 후 object footprint가 goal 안에 있는지 외부 시각으로 판정한다.
- **Existing Asset:** Phase 4 geometry 예정.
- **Missing Work:** success tolerance, object footprint localization, release/retreat sequence.
- **Implementation:** open→retreat→fresh observation→deterministic containment/error 계산을 수행한다.
- **Experiment:** goal 내부/경계/외부 배치와 실제 release trial.
- **Metrics:** release success, containment accuracy, final position error.
- **PASS Gate:** 판정 fixture가 정확하고 trial 결과가 raw image/measurement에서 재현된다.
- **Output Files:** verification code/config/tests, `results/EXP014_runNN/`.
- **Dependencies:** Phase 4, 8, 13 PASS.

## Phase 15 — Closed-Loop Correction

- **Purpose:** 시간 여유가 있을 때 tolerance 밖 결과를 재관찰·재배치한다.
- **Existing Asset:** Phase 14 결과 예정.
- **Missing Work:** retry limit, correction policy, loop safety와 연구 일정 승인.
- **Implementation:** 최대 횟수가 제한된 observe→re-grasp→correct→verify loop만 추가한다.
- **Experiment:** 의도적 offset 배치의 correction trial.
- **Metrics:** correction success, residual error, added time, retry count.
- **PASS Gate:** core system 완료 후 correction이 error를 악화시키지 않고 안전 제한을 지킨다.
- **Output Files:** optional correction code/config/results.
- **Dependencies:** Phase 14 PASS; 일정상 optional.

## Phase 16 — End-to-End Experiment

- **Purpose:** 자연어부터 final visual verification까지 전체 연구질문을 검증한다.
- **Existing Asset:** PASS한 모든 선행 모듈 예정.
- **Missing Work:** frozen protocol, randomized trial matrix, failure taxonomy, analysis script.
- **Implementation:** 검증된 모듈만 고정 버전/config로 연결하고 trial마다 전체 provenance를 저장한다.
- **Experiment:** 정의된 자연어 지시, target, start, goal 조합의 반복 trial.
- **Metrics:** task success, final object position error, total task time, failure cause.
- **PASS Gate:** 사전 등록한 trial 수와 성공/error 기준을 충족하고 모든 결과를 raw data에서 재계산한다.
- **Output Files:** frozen config, end-to-end runner, `results/EXP016_runNN/`, 논문 표/그림.
- **Dependencies:** Phase 0~14 PASS; Phase 15는 optional.
