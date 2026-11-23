# Research Scope

## Research Question

고정 3인칭 카메라 환경에서 VLM의 semantic understanding과 결정론적 metric geometry를 결합해 자연어 object-repositioning 지시를 실제 메카넘 로봇의 검증 가능한 grasp-and-transport task로 수행할 수 있는가?

## Current Development Priority

현재는 전체 연구 목표를 유지하면서 `robot-only path planning → static obstacles → dynamic replanning → real robot execution → manipulation` 순서로 개발한다. STEP 14의 obstacle은 planning software 검증용 `DEVELOPMENT_SYNTHETIC` geometry이며 VLM perception 결과가 아니다.

## Contributions

1. semantic target bbox를 undistortion과 homography로 실제 workspace X/Y에 연결한다.
2. 자연어 지시를 target, destination, approach pose, feasibility-validated path로 변환한다.
3. 외부 카메라 pose feedback으로 물체를 운반하고 최종 배치를 다시 시각적으로 검증한다.

## Included Scope

- Qwen3-VL 기반 target object와 predefined zone 의미 식별
- camera calibration 재사용 및 독립 검증
- bbox reference point → undistorted pixel → world X/Y
- ArUco 기반 robot workspace `(x, y, yaw)`와 robot-center offset
- structured task와 unified workspace state
- known object geometry/config 기반 approach pose
- deterministic boundary/collision/clearance/goal validator
- 2D occupancy grid + A* baseline 또는 검증된 기존 ROS2 기능 재사용
- external-vision feedback mecanum control
- 전면 simple gripper의 open/close, grasp, transport, release
- raw-data 기반 정량 평가와 final visual verification

## Excluded Scope

- VLM 학습/fine-tuning과 foundation model 연구
- 신규 SLAM, Nav2, ArUco detector 또는 multi-marker fusion 알고리즘
- Frappe/ArUco Nano 비교, MCP 자체 연구
- 6-DOF arm, 범용 grasping, force control
- reinforcement learning, end-to-end neural control
- multi-robot, outdoor navigation, 3D SLAM
- event-triggered VLM과 dynamic obstacle 연구

## Assumptions

- 외부 카메라와 workspace는 실험 중 움직이지 않는다.
- 대상과 robot은 평면 바닥에서 움직인다.
- zone은 config의 고정 polygon이며 VLM이 metric polygon을 생성하지 않는다.
- 대상 종류, geometry, grasp interface와 offset은 사전에 알려져 있다.
- 정적 장애물과 명시된 footprint로 feasibility를 판단할 수 있다.
- 카메라 failure 또는 stale localization에서는 정지할 수 있다.

## Hardware

- Ubuntu 22.04 laptop
- fixed USB webcam, nominal calibration 1920×1080
- Jetson Orin Nano, ROS2 Humble
- 4-wheel Mecanum robot, encoder, IMU
- front-mounted simple 1-DOF gripper/lock
- top-mounted ArUco marker

정확한 camera pose, workspace 크기, robot/object/gripper 치수, marker 크기/offset, 속도와 margin은 Phase 0에서 실측·고정해야 하며 현재 `UNKNOWN`이다.

## Object and Gripper Assumptions

- 논문 실험은 하나 이상의 predefined rigid carrier type에 한정한다.
- grasp point와 approach offset은 object config에 정의한다.
- gripper는 open/close 상태와 holding 여부만 제공하면 된다.
- object + robot 결합 footprint는 transport planning에서 별도로 사용한다.
