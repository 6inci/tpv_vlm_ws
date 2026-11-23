# VLM Reposition Research Workspace

고정 3인칭 카메라와 Vision-Language Model을 결합해 자연어로 지정한 물체를 메카넘 이동로봇이 grasp·transport하고, 최종 위치를 외부 시각으로 검증하는 학술대회 연구 Workspace다.

## 시스템 흐름

```text
Natural Language → Task Parsing → VLM Semantic Grounding
→ Undistortion / Homography → Workspace State
→ Grasp Pose / Feasibility Validation → Path Planning
→ Mecanum Control → Grasp / Transport / Release
→ External Visual Verification
```

VLM은 객체와 zone의 의미를 식별한다. 실제 좌표, 충돌 여부, 경로, motor command는 결정론적 모듈이 계산한다.

## 하드웨어

- Laptop: USB 고정 카메라, geometry, Qwen, planning, 시각화, 실험 기록
- Jetson Orin Nano: ROS2 Humble, motor/encoder/IMU/EKF, `cmd_vel`, gripper, emergency stop
- Robot: 4-wheel Mecanum, 전면 1-DOF gripper 또는 locking mechanism
- Robot-mounted camera: 없음

## 현재 상태

- Workspace 문서와 디렉터리만 초기화됨.
- Phase 1 Independent Geometry Validation이 첫 개발 gate임.
- 기존 calibration과 homography는 참조 자산이며 현재 설치 상태의 독립 기준점으로 아직 PASS하지 않음.
- 실행 가능한 통합 pipeline은 아직 없음.

세부 상태는 [PROJECT_STATUS.md](docs/PROJECT_STATUS.md), 전체 순서는 [ROADMAP.md](docs/ROADMAP.md), 현재 단일 작업은 [NEXT_STEP.md](docs/NEXT_STEP.md)를 따른다.

## 구조

```text
vlm_ws/
├── AGENTS.md
├── README.md
├── docs/
├── config/
├── src/vlm_reposition/
│   ├── camera/ geometry/ localization/ perception/
│   └── world/ task/ planning/ control/ evaluation/
├── scripts/
├── tests/
├── datasets/{raw,processed,annotations}/
├── results/
└── legacy_reference/
```

`legacy_reference/`에는 전체 저장소를 복사하지 않는다. 참조 자산은 원래 절대경로와 provenance로 기록한다.

## 참조 자산

- `/home/binci/camera_calibration_1920x1080`
- `/home/binci/qwen_vision_test`
- `/home/binci/mecanum_ros2`

모두 읽기 전용으로 취급한다. 상세 inventory와 한계는 [PROJECT_STATUS.md](docs/PROJECT_STATUS.md)에 기록되어 있다.

## 실행

아직 실행할 구현이 없다. Phase 1 입력과 물리 기준점을 준비한 뒤 검증 코드를 구현한다. 실제 로봇은 자동 실행하지 않는다.

