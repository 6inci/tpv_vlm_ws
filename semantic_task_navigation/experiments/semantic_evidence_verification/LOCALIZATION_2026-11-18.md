# Qwen의 의미 목적지 → 이미지 좌표 후보 영역 (E1 L2, 2026-11-18)

명령 "빨간 큐브를 출하할 수 있는 적절한 장소로 옮겨줘." / 사진 `undistorted_25070552942982.png`. **모델은 새로 돌리지 않았다**(저장된 Qwen 응답과 저장된 Grounding DINO 검증 결과만 재사용). v0.2 코드·동결 프롬프트·기존 결과는 수정하지 않았다(v0.2 지문 `254cfb31da7f…` 동일).
실행 기록: `results/20261118_231429_bay_localization/` (이전 두 번의 실행은 `results/superseded_bay_localization/`: 첫 실행은 자유 바닥 면적을 다각형 채움 값(8,424 px)으로 표기, 둘째는 범례 한 줄이 잘림. 세 실행 모두 구획 좌표는 동일).

## 한 일

| 단계 | 내용 | 출처 |
|---|---|---|
| Qwen 제안 | `area_5` "loading dock", 근거 truck·cones·barriers, 상자 [1248, 0, 1824, 280.8] px (1920×1080). 같은 응답이 L2 탐색 실험 E1에도 그대로 저장됨(원문 일치 확인) | 저장된 원문 `…_203058_46d8/vlm_proposal/raw_response.txt` |
| 객체 | 트럭 1대, 콘 6개 (Grounding DINO + Qwen 크롭 확인, 재사용) | 저장된 탐지 |
| 구체화 (`bay_localization.py`) | Qwen 상자 안에서 어두운 매트 위의 흰 점선 도장을 찾음 → 점선 2개를 직선으로 맞춤 → 교점을 꼭짓점으로 하는 평행사변형 | 사진 픽셀 + 트럭·콘 상자(제외용). 사람이 준 좌표 없음 |
| 자유 바닥 | 위 구획에서 트럭·콘 상자를 뺀 부분 (v0.2 `placement_region` 재사용, 기하만) | 위와 동일 |

## 결과 (CONFIRMED = 이 사진에서 계산됨)

- 상태 `FLOOR_MARKING_ESTABLISHED`. 점선 17개 검출, 선 2개(11개·6개), 사잇각 77.5°, 검사 6/6 PASS.
- 구획 꼭짓점 (px, 보정된 원본 좌표): (1426.7, 172.5) · (1680.3, 168.0) · (1653.4, 56.2) · (1399.7, 60.7), 면적 28,696 px.
- Qwen 상자(161,741 px)의 17.7 %. 상자 안의 콘 6개는 모두 구획 **밖**(0/6, 가장 가까운 콘 96 px). 상자의 나머지 82.3 %는 구획 밖의 바닥·매트 밖 회색 바닥이다.
- 트럭 상자는 구획과 55 % 겹침(상자가 트럭의 키를 포함해 위로 구획 밖까지 나감). 자유 바닥은 상자 제외 후 8,377 px(여유 8 px 후 2,580 px).
- 시각 확인(AI 예비 검토): 사진의 흰 점선 L자 구획과 계산된 구획의 아래·왼쪽 변이 겹친다. 정답 좌표가 없어 IoU·정확도는 계산하지 않았다.

## 아직 검증되지 않은 것 (UNKNOWN / INFERRED)

1. 구획의 **위·오른쪽 변은 그려진 선이 없다**. 점선 두 줄의 끝에서 평행사변형으로 닫은 추론(INFERRED)이다.
2. 직선 평행사변형: 바닥 면의 원근은 모델링하지 않았다(검증된 호모그래피 없음).
3. 임계값은 이 사진을 보면서 정했다. 민감도(`sensitivity.json`): 9가지 중 7가지에서 같은 구획, 흰색 밝기 기준(V)을 185에서 195 이상으로 올리면 점선을 못 찾아 `FLOOR_MARKING_NOT_ESTABLISHED`가 된다.
4. 같은 세션 사진 8장에서 8/8 성립, 꼭짓점 최대 이동 3.1 px. 같은 배치이므로 **반복성**이지 일반화가 아니다(E1의 탐색 창·객체 상자를 그대로 사용).
5. 트럭 탐지 상자는 트럭 바닥 발자국이 아니다. 큐브를 놓을 곳이 바닥인지 트럭 적재함인지, 놓는 면과 그리퍼 도달 여부는 측정되지 않았다. 이는 배치 결과가 아니라 후보 영역이다.
6. v0.2 `placement_region`은 외곽선만 유지하므로 구획 안쪽에 완전히 들어간 물체는 구멍으로 남지 않는다. `hole_check`로 감지한다(E1은 해당 없음: 트럭 상자가 구획 위쪽 경계를 가로지름).
7. 지도와 사진은 어긋남(MISALIGNED, 이동 약 (−32.8, −6.5) px) → 경로계획으로 넘기지 않았다. A*·ROS2·주행은 실행하지 않았다.

## 재현

```bash
cd /home/binci/vlm_ws/semantic_task_navigation
PY=/home/binci/miniconda3/envs/stnav_gui/bin/python; export PYTHONDONTWRITEBYTECODE=1
$PY -m pytest -q experiments/semantic_evidence_verification/test_bay_localization.py   # 11건, 모델·GPU 불필요
$PY experiments/semantic_evidence_verification/localize_bay.py                          # 저장된 결과만 읽고 새 폴더에 기록
```
