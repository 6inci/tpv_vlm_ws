# 공유 실험 환경 사진 15장과 카메라 자세 (env2_20261119 좌표)

- 상태: **TESTING** (자세는 계산으로 검증했고, 사진을 만든 쪽의 확인은 아직 없음)
- 만든 날: 2026-11-22. 기존 폴더와 `~/Downloads` 원본은 읽기만 했다.

## 사진

- 원본은 `~/Downloads/맵사진_전달(2차)/전달/20261120_HHMMSS_view.png` 15장이다. 촬영 시각은 2026-11-20 21:57–22:06이고, 1920×1080 PNG다.
- `photos/`에 원본을 그대로 복사했다. 해시는 `SHA256SUMS`에 있다. 2026-11-22 대화에 첨부된 15장과 sha256이 모두 같다.
- 사용자 설명으로는 팀원과 맞춘 실험 환경이다.
  - 사진마다 빨간 큐브가 올라간 작은 팔레트 하나와 빈 작은 회색 팔레트 하나가 노란 선 밖에 있고, 위치는 사진마다 다르다. 로봇은 모든 사진에서 같은 자리에 있다(AI가 본 것, 확인 안 됨).
  - 사진별 정답 위치는 아직 적지 않았다.
- 15장은 서로 1 px 안에서 겹친다(위상 상관). 그래서 카메라 자세 하나를 15장 모두에 쓴다.

## 카메라: 이 사진들은 `newK = K`로 편 사진이 아니다

| 가정 | 카메라 높이 | 기준점 바닥 오차 최대 (자세에 쓴 10점) |
|---|---|---|
| `newK = K` (기존 8장, 팀원 등록 사진과 같은 방식) | 2.495 m | 4.26 cm |
| 원본(왜곡 남음) | 2.416 m | 4.35 cm |
| **`newK = getOptimalNewCameraMatrix(K, dist, (1920,1080), alpha=0)`** | **2.217 m** | **1.02 cm** |
| alpha 0.5 / 1 | 2.052 / 1.881 m | 2.06 / 3.50 cm |

- alpha 0일 때 결과가 팀원 등록 자세와 거의 같다.
  - 카메라 높이: 2.217 m (등록 2.216 m)
  - 기준점별 오차: 등록 기록과 0.1 cm 안에서 같다. A 0.23, C 1.00, D 1.00, B 3.65 cm (B는 팀원 코드가 자세 계산에서 뺀다).
  - 위치: 등록 자세와 0.79 cm, 0.26° 차이다.
  - **해석(INFERRED): 팀원 등록 때와 같은 카메라이고, 사진만 alpha 0 `newK`로 폈다.** 사진을 만든 쪽의 확인은 받지 않았다.
- 같은 floor homography에서 다시 왜곡을 펴면 맞는 점이 925개에서 343개로 줄었다. 즉 사진은 이미 편 상태다.
- 기존 8장(`5calibration.png` 등)은 `newK = K`이고, 이 사진들과 카메라가 3.8 cm, 0.9° 다르다. **두 묶음에 같은 K와 자세를 쓰면 안 된다.**
- 사용법:
  ```python
  p2m.pixel_to_map(px, rvec, tvec, K_photo, None, z)
  ```
  `rvec`, `tvec`, `K_photo`는 `camera_pose/photoset_20261120_pose.json`의 값을 쓴다. `calibration_result.json`의 K를 쓰면 최대 약 4 cm 틀린다.

## 자세를 구한 방법 (`fit_photoset_pose.py`, 설정 `config.yaml`)

1. 팀원 등록 사진 `undistorted.png`(`newK = K`, 기준점 픽셀이 알려져 있음)와 이 사진 사이의 바닥 homography를 구했다.
   - 방법: SIFT와 MAGSAC
   - 결과: 931/1361 점이 맞았고 중앙 잔차는 0.93 px다.
2. 팀원 기준점 12개의 픽셀을 이 homography로 옮겼다. 손으로 찍은 점이 아니다. 기준점 정의(테이프 가장자리 직선의 교점)는 팀원 것을 그대로 따른다.
3. 팀원 코드 `pixel_to_map.fit_pose`로 다섯 가지 K 가정을 모두 풀었다. 그중 자세에 쓴 점의 최대 오차가 가장 작은 가정을 골랐다.
4. 통과 기준(gate):
   - 자세에 쓴 점의 오차 ≤ 1.1 cm: 1.02 cm
   - 지도 → 픽셀 → 지도 왕복 ≤ 0.01 cm: 0.0 cm
   - 카메라 높이 1.8–2.6 m: 2.217 m
   - 모두 통과
- 팀원 코드는 `/home/binci/vlm_ws/nl_to_route_20261122/vendor/teammate_extra_20261120/`에 바꾸지 않고 복사해 둔 것을 쓴다(`CHECKSUMS.sha256` 확인 OK).

```bash
cd /home/binci/vlm_ws/env2_photos_20261120
/home/binci/miniconda3/envs/stnav_gui/bin/python -I -B fit_photoset_pose.py           # 다시 계산 (camera_pose/ 덮어씀)
/home/binci/miniconda3/envs/stnav_gui/bin/python -I -B fit_photoset_pose.py --check   # 저장된 자세 검사만
```

## PASS에 필요한 것

- 이 `_view.png`를 어떤 K로 폈는지 사진을 만든 쪽(팀원 뷰어로 추정)에서 확인받기.
- 또는 사진에서 위치를 아는 점 하나 이상(예: 줄자로 잰 바닥 표시)을 이 자세로 바꿔 오차를 재기.

## 파일

| 파일 | 내용 |
|---|---|
| `photos/`, `SHA256SUMS` | 사진 15장 (원본 복사) |
| `config.yaml` | 경로, homography 설정, K 가정, 통과 기준 |
| `fit_photoset_pose.py` | 자세 계산과 검사 |
| `camera_pose/photoset_20261120_pose.json` | `K_photo`, `rvec`, `tvec`, 기준점 픽셀, 점별 오차, 다섯 가정의 결과, 구역 자리 픽셀 |
| `camera_pose/overlay_env2_on_photoset.jpg` | 이 자세로 env2 지도 경계(하늘색), 기준점, 구역 자리를 사진에 그린 것 |
| `camera_pose/fit.log` | 실행 출력 |
