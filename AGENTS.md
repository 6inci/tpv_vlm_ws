# Project Rules

## 연구 목표

자연어 지시를 고정 3인칭 카메라의 의미 인식과 결정론적 metric geometry로 grounding하고, 4륜 메카넘 로봇이 단순 전면 gripper로 물체를 이동한 뒤 외부 시각으로 결과를 검증한다. 목표는 코드 양이 아니라 재현 가능한 실험 데이터로 논문의 주장을 검증하는 것이다.

`/home/binci/vlm_ws`를 이 연구의 Source of Truth로 사용한다.

## 연구 범위

- 포함: Qwen3-VL 기반 객체/영역 의미 식별, 왜곡보정과 homography, ArUco robot pose, workspace state, task parsing, known-geometry grasp pose, 결정론적 feasibility validation, 2D planning, 메카넘 제어, 단순 gripper, 외부 시각 검증.
- 제외: foundation model 학습/fine-tuning, 신규 SLAM/Nav2/ArUco 알고리즘, 범용 grasping, force control, RL/end-to-end control, multi-robot, outdoor/3D SLAM, dynamic obstacle 연구.
- 새 기능은 현재 연구질문 또는 PASS gate에 필요할 때만 추가한다.

## 역할 경계

- VLM은 semantic label과 bbox를 출력하며 metric 좌표, 경로, 충돌, motor command를 만들지 않는다.
- Vision geometry는 raw pixel, undistorted pixel, world coordinate를 구분하여 변환한다.
- Task parser는 구조화된 task만 만들고 motor command를 만들지 않는다.
- Planner와 validator는 raw image보다 검증된 workspace state를 사용한다.
- Controller는 검증된 목표/경로만 `vx`, `vy`, `omega`로 실행한다.

## 참조 저장소 보호

다음 경로는 READ-ONLY reference다.

- `/home/binci/camera_calibration_1920x1080`
- `/home/binci/qwen_vision_test`
- `/home/binci/mecanum_ros2`

삭제, 이동, 초기화, history 변경, 대규모 refactor, working code overwrite를 하지 않는다. 필요한 최소 코드나 설정만 이 Workspace로 port하고 출처를 문서화한다. 원본 수정이 필요하면 이유, 영향, rollback을 먼저 설명하고 사용자 승인을 받는다.

## 단계별 개발

1. 기존 코드와 모든 관련 호출 경로를 조사한다.
2. 확인된 상태와 누락 항목을 기록한다.
3. 최소 설계를 정하고 한 Phase만 구현한다.
4. unit/offline test 후 필요한 경우에만 사용자가 hardware test를 실행한다.
5. raw data에서 metric을 자동 계산한다.
6. PASS gate 판정과 문서를 함께 갱신한다.

선행 Phase가 PASS가 아니면 이에 의존하는 다음 Phase는 구현하지 않는다. 설계 문서화는 허용한다. 상태값은 `NOT_STARTED`, `DESIGN`, `IMPLEMENTING`, `TESTING`, `PASS`, `FAIL`, `REWORK`, `FROZEN`만 사용한다. 코드가 존재한다는 이유로 PASS로 표시하지 않는다.

## 재현성

- 수치를 수동 생성하지 않는다. 모든 논문 metric은 저장된 raw data에서 재계산 가능해야 한다.
- 각 run에 timestamp, experiment ID, run ID, git commit, config snapshot, hardware condition, input dataset, parameters, raw/processed measurements, metrics, figures, logs를 남긴다.
- `results/EXPxxx_runNN/`처럼 새 run을 만들고 기존 결과를 덮어쓰지 않는다.
- 사실은 `CONFIRMED`, 근거 있는 해석은 `INFERRED`, 미검증 사항은 `UNKNOWN`으로 구분한다.
- 좌표계, 단위, 왜곡 여부, timestamp를 데이터와 코드에서 명시한다.

## 안전

- 기본 검증 순서는 offline → simulation → visualization → dry-run이다.
- 실제 robot motion은 사용자가 명시적으로 실행할 때만 허용한다.
- camera failure, marker lost/stale, invalid state, planner/communication failure, exception, emergency stop 시 기본 명령은 `cmd_vel = 0`이다.
- OS, kernel, driver, CUDA, ROS2, network 또는 package 설치 변경은 이유, 영향, rollback을 설명하고 승인받기 전에는 하지 않는다.

## 코드 및 문서

- 기존 함수, 변수, topic, TF frame, dependency를 우선 재사용하고 최소 diff로 작업한다.
- calibration/model/camera path, threshold, 물리 치수는 source에 hard-code하지 않고 `config/`에서 관리한다.
- VLM 원문 응답을 보존하고 schema를 검증한 뒤 사용한다. 결정론적 계산은 Python에서 수행한다.
- 새 비자명 로직에는 가장 작은 실행 가능한 검사를 하나 남긴다.
- 구현 상태가 바뀌면 `PROJECT_STATUS.md`, 현재 단일 작업은 `NEXT_STEP.md`, 결정은 `DECISIONS.md`, 실험은 `EXPERIMENTS.md`에 반영한다.
