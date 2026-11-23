# COORDINATE FRAMES

모든 박스, 점, 다각형 데이터에는 좌표계를 함께 기록한다(`coordinate_frame` 또는 `frame` 필드).

| 이름 | 정의 | 사용처 |
|---|---|---|
| `raw_image_px` | 디코딩한 입력 프레임 그대로. 왜곡 보정 전. 원점은 좌상단, x는 오른쪽, y는 아래쪽. 단위는 px. 이미지는 EXIF 회전을 무시하고 읽는다(`IMREAD_IGNORE_ORIENTATION`) | 모든 검출, 관계, 지도, 접근 위치, 경로 |
| `undistorted_image_px` | `cv2.undistort(K, dist, newK=K)` 결과 | UNDISTORTED **보기 전용**. 계산에는 쓰지 않음 |
| `roi_local_px` | ROI 크롭을 균일 배율 `scale`로 확대한 이미지 | 큐브 후보 검출 |
| `grid_row_col` | 계획 격자 셀 (r, c). 셀 크기는 `cell_px` | A*, Dijkstra, 유효 셀 |
| `ground_map_xy` | 바닥 평면 metric 좌표 | **비활성**. 검증된 homography가 없음 |

## 변환식

```text
roi_local_px → raw_image_px :  x_raw = crop_x0 + x_roi / scale,  y_raw = crop_y0 + y_roi / scale
raw_image_px → roi_local_px :  x_roi = (x_raw − crop_x0) * scale
grid_row_col → raw_image_px :  셀 중심 (x, y) = ((c + 0.5)·cell_px, (r + 0.5)·cell_px)
raw_image_px → grid_row_col :  (r, c) = (floor(y / cell_px), floor(x / cell_px))
Qwen grounding 출력 → raw_image_px : 0–1000 정규화 값 × (W, H) / 1000
검증 크롭 : crop_xyxy_raw와 crop_scale을 기록. 녹색 박스는 같은 식으로 그림
```

- 각 실행의 `target_inspection.json`에는 ROI마다 `transform`(`crop_xyxy_raw`, `scale`, `formula`)이 저장된다.
- 테스트(`tests/test_target_inspection.py`)에서 확인한 내용
  - 확대 ROI에서 찾은 박스를 원본 좌표로 되돌리면 합성 정답과 1 px 이내로 일치한다.
  - 왕복 변환 오차는 1e-9 이하다.

## 보정 호환성 판정 (`geometry/calibration.py`)

- 보정 파일: `/home/binci/camera_calibration_1920x1080/calibration_result.json`
  - HD Web Camera, 1920×1080, RMS 0.206 px, 2026-08-14
  - 읽기만 한다.
- 왜곡 보정 보기를 허용하는 조건
  1. 프레임 해상도가 보정 해상도와 같다.
  2. 카메라 모델이 같다(이미지: EXIF Model / 라이브: V4L2 장치 이름).
- 위 조건을 만족해도 **현재 설치 상태에서 내부 파라미터가 유효한지는 검증되지 않았다**.
- metric 출력(`CALIBRATED_GROUND_MAP`)에 필요한 조건
  - 내부 파라미터가 현재 설치 상태에서 검증되었을 것
  - 현재 바닥 배치용 Homography가 `validated` 상태일 것
- 현재 상태
  - 둘 다 충족되지 않아 항상 `IMAGE_SPACE_DEMO`로 동작한다.
  - 2026-09-06 Homography는 이전 배치용이라 사용하지 않는다.
  - v0.1에는 지상 좌표 계획이 구현되어 있지 않다. 조건이 충족되어도 그 사실을 기록한 뒤 IMAGE_SPACE_DEMO로 실행한다.

## 단위와 가정 (`config/planning.yaml`)

아래 값은 모두 **이미지 픽셀 단위 가정값**이며 물리량이 아니다.

| 항목 | 값 |
|---|---|
| `robot_radius_px` | 35 |
| `safety_margin_px` | 40 |
| `cell_px` | 8 |
| 픽업 도달 거리 | 140 px |
| 하역 도달 거리 | 180 px |
| `relations.near_px` | 150 |

참고 이미지에는 원근과 렌즈 왜곡이 있어서, 같은 픽셀 거리라도 화면 위치에 따라 실제 거리가 다르다.
