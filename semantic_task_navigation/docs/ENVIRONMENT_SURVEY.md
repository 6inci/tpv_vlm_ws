# 환경 조사 기록 (프로젝트 준비 단계)

- 조사일: 2026-11-15 (Asia/Seoul)
- 범위: 조회만 했다. 설치, 삭제, 업그레이드, 기존 파일 수정은 하지 않았다.
- 표기:
  - `CONFIRMED`: 이번 조사에서 직접 확인함
  - `INFERRED`: 근거 있는 해석
  - `UNKNOWN`: 확인하지 못함

## 1. 요약

| 항목 | 결과 |
|---|---|
| 프로젝트 폴더·구조 | CONFIRMED, 생성 완료 |
| 참고 이미지 | READY, 출처 CONFIRMED (§2) |
| Qwen3-VL-4B / Grounding DINO 가중치 | CONFIRMED, 로컬 캐시에 완전하게 존재 |
| 모델 실행 환경 | CONFIRMED: `qwen_vl`에서 CUDA 동작 |
| PySide6 | CONFIRMED: 모든 환경에 미설치 |
| 현재 카메라 설치 상태의 metric geometry | UNKNOWN (§6.3) |
| 전체 구현 준비 상태 | `PARTIALLY_READY` (§9) |

## 2. 참고 이미지

### 2.1 출처 판단

1. 첫 탐색(약 21:08)
   - 다운로드 폴더에 `warehouse_scene_reference_01.jpg`와 `2026-11-14-182900(1).jpg`가 없었다.
   - 업로드 원본명과 촬영 시각이 일치하는 `/home/binci/Pictures/Webcam/2026-11-14-182900.jpg`를 복사했다.
     - 업로드 원본명 `2026-11-14-182900(1).jpg`에서 `(1)`을 빼면 이 파일명이 된다.
     - EXIF DateTime은 `2026:11:14 18:29:00`이다.
2. 작업 중 확인 (CONFIRMED)
   - `/home/binci/Downloads/warehouse_scene_reference_01.jpg`가 2026-11-15 21:09:14에 생성됐다.
   - 이 파일의 SHA-256은 배치한 파일과 같은 `d1984a19…`다. 크기도 429,599 bytes로 같다.
   - **ChatGPT에서 받은 파일은 Webcam 원본과 바이트 단위로 동일하다.** 배치한 파일은 덮어쓰지 않고 그대로 두었다.
3. 메타데이터와 내용 (CONFIRMED)
   - EXIF Model은 `HD Web Camera`, Software는 `cheese 41.1`이다.
   - 내용은 물류·공정 모형 작업장이다. 로봇팔, 랙, 팔레트, 드럼, 트럭, 콘이 보인다.
   - 기존 실험 EXP110 pilot 이미지 `P06_cheese_2026-11-14-182900.jpg`와 SHA-256이 같다.
4. 사용하지 않은 파일: `/home/binci/Downloads/warehouse_robotics_1920x1080.png`. 이름은 비슷하지만 9월 29일에 만든 다른 PNG다.

### 2.2 검증 결과 (CONFIRMED)

검증 환경: `qwen_vl` Python, Pillow 12.3.0, OpenCV 5.0.0.

| 항목 | 값 |
|---|---|
| 경로 | `assets/reference/warehouse_scene_reference_01.jpg` |
| 형식 | JPEG (SOI `FFD8FF`, EOI `FFD9`, Pillow format=`JPEG`, RGB) |
| 해상도 | 1920×1080. OpenCV shape `(1080, 1920, 3)`, uint8 |
| 크기 | 429,599 bytes |
| SHA-256 | `d1984a19d56443e5844a4f9ba1f5ff77e3349e0d4b8d289496a2b7ef6a6bb439` (원본과 동일) |
| 로드 | Pillow 전체 디코드 OK, `cv2.imread` OK |
| 좌표 공간 | raw (왜곡 보정 전) 픽셀 |

## 3. 시스템 (CONFIRMED)

| 항목 | 값 |
|---|---|
| OS | Ubuntu 22.04.5 LTS, kernel 6.8.0-138-generic |
| 시스템 Python | `/usr/bin/python3` 3.10.12 |
| GPU | NVIDIA GeForce RTX 5070 Ti Laptop GPU, compute capability 12.0 |
| VRAM | 12,227 MiB. 조사 시점 사용량 1,733 MiB (데스크톱 등) |
| 드라이버 | 580.178.04 (드라이버 기준 CUDA 13.0) |
| CUDA toolkit | `nvcc` 없음. PyTorch wheel 내장 CUDA 런타임을 사용한다 |
| ROS2 | Humble (`/opt/ros/humble`) |

`~/.bashrc`는 ROS2 Humble과 `~/mecanum_rviz_ws/mecanum_ros2/install`을 source한다. 그래서 터미널의 `PYTHONPATH`에 ROS 경로가 들어간다.

## 4. Python / Conda 환경 (CONFIRMED)

Conda 26.3.2, `/home/binci/miniconda3`.

| 환경 | Python | torch (CUDA) | transformers | OpenCV | 기타 주요 패키지 | 용도 |
|---|---|---|---|---|---|---|
| `qwen_vl` | 3.10.20 | 2.13.0+cu130 (13.0, 사용 가능) | 5.16.1 | opencv-python 5.0.0 (Qt5 highgui) | accelerate 1.14.0, qwen-vl-utils 0.0.14, numpy 2.2.6, Pillow 12.3.0, scipy, networkx, PyYAML | Qwen3-VL-4B, Grounding DINO, Florence-2 (EXP110/111) |
| `qwen_vl_q4` | 3.10.20 | 2.13.0+cu130 | 5.16.1 | opencv-python 5.0.0 (Qt5) | `qwen_vl` + bitsandbytes 0.50.2 | Qwen3-VL-8B 4-bit |
| `yolo_world` | 3.10.20 | 2.13.0+cu130 | 5.16.1 | opencv-python 5.0.0 (Qt5) | ultralytics 8.4.164, matplotlib | YOLO-World, YOLOE |
| `env_isaaclab` | 3.11.15 | 2.7.0+cu128 | 4.57.6 | headless 4.11.0 | pytest 9.1.1 | Isaac Lab (이 프로젝트와 무관) |
| `base` | 3.13.13 | 없음 | 없음 | 없음 | | |
| 시스템 `/usr/bin/python3` | 3.10.12 | 없음 | 없음 | 4.5.4 | PyQt5 5.15.6, pytest 6.2.5, numpy 1.21.5 | ROS2 Humble |
| `~/qwen_vl_env` (venv) | 3.10.12 | 없음 | 없음 | 없음 | | 비어 있음, 사용 안 함 |

- 환경 경로 규칙: `/home/binci/miniconda3/envs/<env>/bin/python`
- `qwen_vl` 계열 3개 환경에는 pytest가 없다.
- 세 환경의 `opencv-python`은 `cv2/qt` 플러그인(Qt5)을 번들한다.
  - 같은 프로세스에 PySide6(Qt6)를 함께 쓰면 Qt 플랫폼 플러그인 충돌이 알려져 있다.
  - 이 충돌은 이번에 재현하지 않았다 (INFERRED).

## 5. 기존 모델 (CONFIRMED)

캐시 루트: `/home/binci/.cache/huggingface/hub`. 아래 표의 모델 모두 `*.incomplete` blob이 0개다.

| 모델 | 캐시 / 경로 | snapshot | 크기 | 실행 환경 | 측정 peak VRAM |
|---|---|---|---|---|---|
| Qwen3-VL-4B-Instruct | `models--Qwen--Qwen3-VL-4B-Instruct` | `ebb281e…` (refs/main) | 8.3 GB | `qwen_vl`, bf16 | 8.915 GiB (EXP110 run03 C1) |
| Grounding DINO tiny | `models--IDEA-Research--grounding-dino-tiny` | `a2bb814…` | 659 MB | `qwen_vl` (transformers 내장 `AutoModelForZeroShotObjectDetection`) | 1.991 GiB (EXP110 run03 C2) |
| Qwen3-VL-4B + DINO 동시 | 위 두 모델 | | | `qwen_vl` | 8.911 GiB (EXP110 run03 C6-dino) |
| Qwen3-VL-8B-Instruct | `models--Qwen--Qwen3-VL-8B-Instruct` | `0c351dd…` (**refs/main 없음**) | 17 GB | `qwen_vl_q4`, 4-bit NF4 | 7.014 GiB (EXP110 x6 M3) |
| Florence-2 base / large | `models--florence-community--Florence-2-{base,large}` | | 447 MB / 1.5 GB | `qwen_vl` | 미기록 |
| SAM2.1 hiera-tiny | `models--facebook--sam2.1-hiera-tiny` | | 149 MB | | 미기록 |
| YOLOE-26s (text / prompt-free) | `/home/binci/weights/yoloe/yoloe-26s-seg{,-pf}.pt` | | 30 / 38 MB | `yolo_world` | 미기록 |
| YOLO-World s | `/home/binci/vlm_ws/new_experiment_environment/results/YOLO_WORLD_INTEGRATION_20260929T012518+0900/07_yolo_world/yolov8s-world.pt` | | 26 MB | `yolo_world` | 미기록 |
| CLIP ViT-B/32 | `/home/binci/weights/clip/ViT-B-32.pt` | | 338 MB | `yolo_world` | |

- peak VRAM 수치의 출처: 기존 결과 JSON에 저장된 `torch.cuda.max_memory_allocated` 값이다. `/home/binci/vlm_ws/new_experiment_environment/my_own_ws/EXP110_word_grounding/results/`
  - CUDA context와 데스크톱 사용량은 이 값에 포함되지 않는다.
- 12 GB GPU에서 가능한 구성
  - ~~4B bf16과 DINO tiny의 동시 상주는 가능하다 (INFERRED).~~ **정정 (2026-11-15 실측, CONFIRMED): 불가능하다.** 두 모델을 모두 올린 뒤 DINO 추론을 하자 CUDA OOM이 났다. 이때 할당량은 9.57 GiB였다. 원래 추론의 근거였던 EXP110 C6-dino 측정은 두 모델을 동시에 올린 상태가 아니었던 것으로 보인다. 그래서 구현에서는 GPU에 한 모델만 올린다(`gpu_residency: exclusive`). 단독 실측치는 Qwen 로드 8.30 GiB, DINO 추론 peak 1.97 GiB였다.
  - 8B bf16(가중치 약 17 GB)은 GPU에만 올릴 수 없다.
- 8B를 쓸 때는 refs/main이 없으므로 `revision="0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"`를 명시해야 한다. 기존 `x6_smoke.py`가 이 방식을 쓴다.
- 기존 로딩 방식: `/home/binci/vlm_ws/new_experiment_environment/my_own_ws/EXP110_word_grounding/tools/run_models.py`
  - `AutoProcessor` / `Qwen3VLForConditionalGeneration.from_pretrained(mid, local_files_only=True, dtype=torch.bfloat16).to('cuda')`
  - 실행 예: `<env python> tools/run_models.py --run <run> --split dev --cond C1`

## 6. 재사용 후보 코드 (경로만 기록, 가져오지 않음)

READ-ONLY 위치:

- `/home/binci/camera_calibration_1920x1080`
- `/home/binci/qwen_vision_test`
- `/home/binci/mecanum_ros2`
- `/home/binci/vlm_ws`의 기존 프로젝트

### 6.1 경로 약칭

| 약칭 | 실제 경로 |
|---|---|
| `NEE` | `/home/binci/vlm_ws/new_experiment_environment/my_own_ws` |
| `WFX` | `/home/binci/vlm_ws/wanna_find_detailed_experimental_x,Y` (경로에 쉼표가 있으므로 인용 필요) |
| `CAL` | `/home/binci/camera_calibration_1920x1080` |
| `QVT` | `/home/binci/qwen_vision_test` |

### 6.2 기능별 후보

| 기능 | 후보 경로 | 비고 |
|---|---|---|
| 이미지 / 카메라 입력 | `NEE/tools/capture_frames.py`, `NEE/EXP110_word_grounding/tools/capture.py`, `CAL/compare_raw_undistorted.py` | 앞의 두 파일은 `CAL/compare_raw_undistorted.py`의 카메라 열기 로직을 port한 것이다 |
| 카메라 설치 이동 확인 | `NEE/EXP111_small_objects/tools/camera_check.py` | 기준 이미지 대비 shift를 `phaseCorrelate`로 측정 |
| 왜곡 보정 | `CAL/undistort_live.py`, `CAL/camera_calibration_opencv.yaml`, `CAL/calibration_result.json` | HD Web Camera 1920×1080, RMS 0.206 px, 2026-08-14 |
| 왜곡 보정 + Homography (legacy 재현) | `WFX/src/vlm_reposition/geometry/legacy.py` | raw → undistorted → world 재현. 소프트웨어 재현만 PASS했다 |
| Homography | `QVT/build_homography12_undistorted.py`, `QVT/homography_12pt_undistorted.json`, `QVT/cross_validate_homography12_undistorted.py`, `QVT/pixel_to_world_rayplane.py` | 2026-09-06 이전 카메라 배치 기준 (§6.3) |
| ArUco pose | `CAL/aruco_pose.py`, `CAL/aruco_pose_id0.py` | |
| Qwen 추론 | `NEE/EXP110_word_grounding/tools/run_models.py`, `WFX/src/vlm_reposition/perception/qwen_grounding.py` | 두 번째 파일은 JSON parsing·schema 검증·bbox 변환·로컬 로딩을 담당한다 |
| 객체 탐지 (DINO / YOLOE / YOLO-World / Florence) | `NEE/EXP110_word_grounding/tools/run_models.py` | 조건 C2–C6 |
| 장면 전용 초안: 객체 | `NEE/2026-11-14_object_registry_draft/object_registry_draft.json` | 이 참고 이미지와 같은 장면 |
| 장면 전용 초안: 영역 | `NEE/2026-11-15_zone_draft/semantic_zones_draft_v2.json` | 같은 장면 |
| 장면 전용 초안: 공간 관계 | `NEE/EXP120_E3a_relation/RELATION_DEFINITIONS.md` | 같은 장면 |
| A* 경로계획 | `WFX/src/vlm_reposition/planning/grid_astar.py` | 8-connected, safety cost layer. offline PASS |
| 충돌 형상 | `WFX/src/vlm_reposition/planning/collision_geometry.py` | |
| 경로 검증 | `grid_astar.py:validate_grid_path`, `dynamic_state.py:validate_current_path`, `replanning.py:validate_replanned_path` (모두 `WFX/src/vlm_reposition/planning/`) | |
| 경로 시각화 | `WFX/scripts/run_path_planning_baseline.py`, `WFX/scripts/evaluate_navigation_path_validity.py` | |
| 결과 overlay 시각화 | `NEE/2026-11-15_zone_draft/make_semantic_overlay.py`, `NEE/EXP110_word_grounding/tools/viz_examples.py` | |
| 실시간 Qwen 데모 | `QVT/realtime_qwen_working.py` | |
| 로봇 (메카넘 / ROS2) | `/home/binci/mecanum_ros2` | 내용 미검토 |

### 6.3 기하 정보의 유효성

| 항목 | 상태 | 내용 |
|---|---|---|
| 내부 파라미터 | UNKNOWN | 2026-08-14 보정값이다. 카메라 모델명(`HD Web Camera`)은 같다. 같은 개체인지, 초점이나 줌이 바뀌었는지는 확인하지 않았다. |
| Homography | 현재 배치에 사용 불가 (INFERRED) | 2026-09-06에 이전 바닥 배치로 만든 것이다. |

- 기존 EXP110/120은 raw 픽셀 공간에서만 실험했다. 관계 거리도 픽셀 단위다.
- 현재 작업장 배치에서 metric 접근 위치와 경로를 만들려면 새 world 기준점이 필요하다.

## 7. GUI (CONFIRMED)

- PySide6: 모든 conda 환경, 시스템 Python, user site 어디에도 없다. 설치하지 않았다.
- 시스템 Python에 PyQt5 5.15.6이 있지만 ROS2 시스템 패키지다. 사용하지 않는 것을 권장한다.
- `qwen_vl` 계열의 `cv2.imshow`는 Qt5 backend로 동작한다.

## 8. 환경 구성 제안

**권장: C (GUI 환경과 모델 추론 환경을 분리)하고, 추론은 B 방식(별도 worker 프로세스)으로 실행한다.**

| 안 | 판단 | 이유 |
|---|---|---|
| A. 기존 모델 환경 하나 사용 | 비권장 | `qwen_vl`에 PySide6를 설치하면 기존 환경이 바뀐다. 이 환경으로 실행한 EXP110 결과는 동결(FREEZE) 상태다. 번들 Qt5(cv2)와 Qt6(PySide6)가 충돌할 위험도 있다. |
| B. 기존 모델 환경을 별도 프로세스로 실행 | 채택 (추론 측) | `qwen_vl`을 수정 없이 재사용한다. 모델을 상주 worker로 띄우면 로드 시간(수십 초)을 한 번만 쓴다. GUI는 모델이 멈추거나 실패해도 함께 죽지 않는다. |
| C. GUI 전용 환경과 모델 추론 환경 분리 | 채택 (전체 구조) | GUI 환경에는 PySide6, `opencv-python-headless`, numpy, pytest 정도만 둔다. 결정론적 geometry·navigation 코드는 numpy/OpenCV만 쓰도록 해서 양쪽 환경에서 동작하게 한다. |

- 구성 형태 (INFERRED):
  - GUI 프로세스(새 환경)
  - ↔ 추론 worker(`qwen_vl` python, 기존 모델 캐시 사용)
  - 둘 사이는 JSON 요청·응답으로 통신한다. stdin/stdout, 로컬 socket, 파일 중 하나를 쓴다.
  - VLM 원문 응답은 worker가 그대로 돌려주고, GUI 쪽에서 schema를 검증한 뒤 `runs/`에 저장한다.
- GPU: ~~4B bf16과 DINO tiny를 한 worker에 같이 올려도 된다~~ → **정정: 동시 상주하면 OOM이 난다(실측). 한 모델씩 교체해서 쓴다.** 8B 4-bit는 별도 worker(`qwen_vl_q4`)로만 실행한다.
- 필요한 조치: 새 GUI 전용 conda 환경을 만들려면 사용자 승인이 필요하다. 이번 단계에서는 만들지 않았다.

## 9. 전체 구현 준비 상태: `PARTIALLY_READY`

### 9.1 구현 전에 결정·해결할 것

1. **실행 환경 승인** (차단 요인)
   - §8 구성을 승인받아야 한다.
   - 그 뒤 GUI 전용 환경을 만들고 PySide6를 설치한다.
   - 승인 전에는 GUI를 실행할 수 없다.
2. **Metric geometry 범위 결정**
   - 다음 둘 중 하나를 정해야 한다.
     - 첫 구현을 raw 픽셀 공간에서 시작하고, metric 변환은 이후 Phase로 미룬다.
     - 지금 바로 현재 배치의 world 기준점과 holdout 점을 측정한다.
   - 기존 내부 파라미터를 재사용할지도 같이 정한다.
3. **테스트 실행기**
   - `qwen_vl`에는 pytest가 없다.
   - GUI 환경에 pytest를 넣고, 모델 의존 코드는 worker 경계 뒤에 두는 방식을 권장한다.
4. **로봇 인터페이스 정보**
   - 아래 정보가 없다 (UNKNOWN).
     - 온보드 카메라 2개의 사양
     - 그리퍼 인계 방식
     - ROS2 topic과 frame
   - 외부 비전 단계에는 필요하지 않다. robot 모듈을 시작하기 전에 필요하다.
5. **`runs/`와 참고 이미지의 버전 관리 정책**
   - Git을 초기화하기 전에 정한다.
   - 후보는 `.gitignore`의 주석에 적어 두었다.

## 10. 보존 확인

- 새로 만들거나 바꾼 파일은 `/home/binci/vlm_ws/semantic_task_navigation/` 내부에만 있다.
  - 그 밖에는 scratchpad 임시 스크립트만 있다.
- 다음 원본은 읽기만 했다.
  - 참고 이미지 원본
  - 기존 연구 코드와 결과
  - `AGENTS.md`
  - 기존 conda 환경
  - ROS2, CUDA, NVIDIA 설정
  - 모델 캐시
- Git 초기화와 원격 저장소 연결은 하지 않았다.
- `/home/binci/vlm_ws`는 Git 저장소가 아니다 (CONFIRMED).

## 11. 구현 단계 갱신 (2026-11-15)

- 환경 구성 C + B를 적용했다.
  - 새 conda 환경 `stnav_gui`를 만들었다: Python 3.10.20, PySide6 6.12.0, opencv-python-headless 5.0.0.93, numpy 2.2.6, PyYAML 6.0.3, pillow 12.3.0, pytest 9.2.0 ([requirements.txt](../requirements.txt)).
  - 기존 `qwen_vl` 환경은 변경하지 않았다. 모델 worker 프로세스로만 사용한다.
  - rollback: `conda env remove -n stnav_gui`
- 기존 환경과의 상호작용 (CONFIRMED)
  - `~/.bashrc`의 ROS2 `PYTHONPATH` 때문에 pytest가 ROS 플러그인(`launch_testing`, `ament_*`)을 자동으로 불러와 실패했다.
  - `pytest.ini`에서 해당 플러그인만 껐다. ROS 환경은 바꾸지 않았다.
  - pip가 출력한 `generate-parameter-library-py requires jinja2` 경고도 같은 `PYTHONPATH` 때문이다. 설치에는 영향이 없다.
- 카메라 (CONFIRMED, 2026-11-15)
  - 연결된 장치는 `/dev/video0 HP True Vision FHD Camera`(노트북 내장)와 `/dev/video2`(IR)다.
  - `/dev/video0`은 1920×1080 MJPG, 약 15 fps로 열렸다. 프레임은 저장하지 않았다.
  - 보정 대상인 외부 `HD Web Camera`는 연결되어 있지 않다.
- 지도
  - 같은 세션의 zone 초안(P01)을 `config/maps/reference_scene_map.json`으로 옮겼다(`DRAFT_UNCONFIRMED`).
  - P01과 참고 이미지(P06)의 위상상관 이동은 (−0.15, −0.42) px, 응답 0.91이다(같은 카메라 자세).
