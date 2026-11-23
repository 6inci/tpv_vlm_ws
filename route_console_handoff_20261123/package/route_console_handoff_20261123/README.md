# 경로 계획 콘솔 시스템 묶음 (2026-11-23T08:18)

자연어 지시 하나와 사진 한 장으로 옮길 대상(A)과 놓을 곳(B)을 고르고, env2 지도 좌표의 정차 자세와 경유점 경로(팀원 경로 형식)를 만드는 시스템의 파일 묶음이다.

- 모든 파일은 원래 PC의 원본을 복사한 것이다. 원본 경로와 sha256은 `MANIFEST.txt`에 있다.
- 시스템 상태: TESTING (실험용, 실제 로봇과 자동 연결 안 됨)

## 폴더

- `1_console/`: 콘솔 본체. 화면(`web/`), 서버, 다섯 단계 계산(`engine.py`), 설정(`config/console.yaml`), 모델 없는 검사(`test_offline.py`), 사용 설명(`README.md`)
- `2_reused/`: 콘솔이 불러 쓰는 기존 코드와 설정. 원래 폴더 이름(`/home/binci/vlm_ws/` 아래) 그대로 두었다.
- `3_results/`: 결과 기록
  - `photos15_this_system/`: 사진 15장 결과
  - `dev8_threshold_photos/`: 기준값을 정한 개발 사진 8장 결과
  - `console_runs/`: 콘솔 실행 예 10건
- `4_design_evidence/`: 설계 근거 문서 (이전 구조 비교, 구성 요소 빼기 시험)
- `5_paper_figures/`: 논문 그림 후보
- `MANIFEST.txt`: 묶음 안 경로, 원본 경로, sha256

## 다섯 단계와 파일

1. **지시 이해**: Qwen3-VL-4B-Instruct가 지시 글자만 읽고 정해진 항목(할 일, 옮길 것, 물체 옆 장소, 출발, 도착지)으로 바꾼다. 사진은 보지 않는다.
   - 프롬프트 `console_slots_v2b`: `1_console/config/console.yaml`
   - 형식 검사, 해석 보정, 보류: `1_console/engine.py`의 `parse_slots`, `task_plan`
2. **장면 인식**: Grounding DINO-tiny가 후보 박스를 내고(노란 선 둘레는 확대해 한 번 더 검출), SigLIP2 so400m이 후보를 예시 이미지(큐브, 빈 팔레트, 짐 실린 팔레트)와 비교한다.
   - `2_reused/context_policy_20261122/extract.py`
   - `2_reused/hybrid_exemplar_20261121/hybrid_pipeline.py`, `config.yaml`
   - DINO 호출: `2_reused/semantic_task_navigation/experiments/multimodel_target_viewer/mm_pipeline.py`의 `MMViewer.photo`
   - DINO 검출어·기준값: `2_reused/semantic_task_navigation/experiments/task_context_2d_navigation/config.yaml`
3. **대상 결정**: Python 규칙
   - 작업 공간: 노란 선 밖 0–15 cm, 5S 금지 구역 밖
   - 큐브: 예시 점수 ≥ 0.88, 짐 실린 팔레트 위(받침 점수 ≥ 0.8619)
   - 빈 팔레트: 점수 ≥ 0.8531, 점유된 것 제외
   - 1·2위 점수 차이 ≥ 0.02
   - "X 옆" = X 구역까지 30 cm 이내
   - 받침·빈 팔레트 기준값은 개발 사진 8장에서 정했다(`data/dev/calibration.json`).
   - 파일: `2_reused/context_policy_20261122/policy.py`, `common.py`, `config/policy.yaml`, `zones/zones_env2.json` (사람이 그린 5S 구역 초안을 env2 지도로 옮긴 것)
4. **좌표 변환**: 박스 중심 → env2 지도(m). 물체 윗면 높이 평면으로 계산한다.
   - 팀원 카메라 모델: `2_reused/nl_to_route_20261122/vendor/teammate_extra_20261120/1_required/pixel_to_map.py`
   - 사진 카메라 행렬: `2_reused/env2_photos_20261120/camera_pose/` (사진은 alpha 0 newK로 편 것, 맞춤으로 추정)
5. **경로 생성**: 그리퍼 끝이 물체를 향하는 정차 자세, 4방향 A*(1 cm 칸), 구간마다 회전 1번
   - `2_reused/nl_to_route_20261122/nl_to_path.py`, `POLICIES.md`
   - 팀원 경로 형식(6항목)으로 저장한 뒤, 팀원 `clearance_check.py`로 다시 검사한다.

## 결과

### 사진 15장 (`3_results/photos15_this_system/`)

- 지시: "빨간 큐브를 비어 있는 팔레트로 옮겨줘."
- A: 맞음 14, 틀림 0, 보류 1 (큐브가 노란 선 밖 16 cm)
- B: 맞음 12, 맞는 보류 1 (빈 팔레트 없음), 놓침 2 (점수 0.835 미달 1, 충전소 기둥과 0.006 차이로 보류 1)
- 경로: 11/15. 팀원 여유 검사 불가 0 cm, 차체 여유 최소 3.0 cm
- 정차 자세:
  - 그리퍼 끝이 물체까지 0.3–8.4 cm 모자람
  - 11장 중 9장에서 정차 한 곳이 옆으로 6.3–7.8 cm 어긋남(기준 2 cm)
- 등급 원자료 (decisions.json): A 놓침(보류) 1, A 맞음 14, B 놓침(보류) 2, B 맞는 보류 1, B 맞음 12
- 조건:
  - 정답 파일 `data/gt_test.json`은 AI가 만든 것으로, 사람이 확인하기 전이다.
  - 이 15장은 규칙을 고칠 때 참고한 사진이라 개발 결과다(독립 시험 아님).

### 기존 시스템과 숫자를 섞지 말 것

- 논문 초안 3.1의 "30건 중 25건 확정"과 팀원에게 준 KRoC 자료(경로 9/15)는 기존 시스템(`nl_to_route_20261122`의 11-22 실행) 결과다.
- 기존 시스템은 SigLIP2 base 시설 확인을 쓰고, 띠 확대 검출과 받침 검사는 쓰지 않는다.
- 같은 15장에서 기존 결과는 다음과 같았다:
  - A: 맞음 13, 틀림 1 (선반 빨간 부품), 보류 1
  - B: 맞음 11, 맞는 보류 1, 놓침 3
- 논문에는 둘 중 한 시스템의 숫자만 쓴다.

### 콘솔 실행 예 (`3_results/console_runs/`, 실제 Qwen 해석)

- `20261122_192459_20261120_215815_view`: "빨간 큐브를 비어 있는 팔레트로 옮겨줘." → PATH_OK (현재위치→큐브파지, 큐브파지→팔레트배치)
- `20261122_192502_20261120_215900_view`: "컨베이어 옆에 있는 물건을 빈 팔레트에 놓아줘." → PATH_OK (현재위치→큐브파지, 큐브파지→팔레트배치)
- `20261122_192504_20261120_220042_view`: "빨간 큐브를 하역장으로 가져가." → PATH_OK (현재위치→큐브파지, 큐브파지→하역장)
- `20261122_192506_20261120_215815_view`: "충전소로 이동해." → PATH_OK (현재위치→충전소)
- `20261122_192509_20261120_215815_view`: "하역장에서 출발해서 컨베이어로 가." → PATH_OK (하역장→컨베이어)
- `20261122_192510_20261120_215815_view`: "큐브 좀 치워줘." → HOLD — 도착 장소 'none'에 정해진 정차 자리가 없음
- `20261122_192512_20261120_215815_view`: "선반에 있는 상자를 트럭에 실어줘." → HOLD — 옮길 물체를 확정하지 못함: 조건을 모두 통과한 후보가 없음
- `20261122_195711_20261120_220344_view`: "하역장 옆에서 빨간 큐브를 챙겨 충전소 옆으로 이동시켜줘" → HOLD — 옮길 물체를 확정하지 못함: 조건을 모두 통과한 후보가 없음. 상위 후보: m59(점수 0.913) 장소 옆: 하역장 구역까지 47 cm (한도 30 cm)
- `20261122_200350_20261120_220632_view`: "하역장 옆에서 챙길 것을 들고 충전소 옆에 회색 팔레트로이동시켜줘" → PATH_OK (현재위치→큐브파지, 큐브파지→팔레트배치)
- `20261122_193058_live_simulated_220632`: "빨간 큐브를 비어 있는 팔레트로 옮겨줘." → PATH_OK (현재위치→큐브파지, 큐브파지→팔레트배치)

- 실행마다 `record.json`(Qwen 원문 응답, 후보 점수, 판단 이유, 좌표, 경로, 검사), `overlay.jpg`, `waypoints.csv`, 팀원 형식 경로 파일이 있다.
- 장소 → 장소 이동의 출발 보정 규칙(`task_plan`)은 이 실행들 뒤(11-22 저녁)에 넣었다.

## 논문 그림 후보 (`5_paper_figures/`)

- `220103_photo.png`: 원본 사진 (환경 그림 바탕)
- `220103_path_overlay_this_system.jpg`, `215826_path_overlay_this_system.jpg`: 인식부터 경로까지. 아래 글자 띠는 잘라서 쓴다.
- `215847_hold_example_policy.jpg`: 빈 팔레트가 없어 B를 보류한 예. 탈락 후보의 이유가 표시돼 있다.
- `vlm_five_stages.pdf`: 다섯 단계 그림

## 들어 있지 않은 것

- 모델 가중치
- 미리 계산한 인식 자료: `context_policy_20261122/data/test` 1.7 GB, `data/dev`의 임베딩 파일
- 사진 원본: 220103 한 장만 포함
- 실행 환경, v0.2 모델 작업 프로세스(`semantic_task_navigation` 공통 코드)
- 코드 안 경로가 원래 PC의 절대 경로라서, 이 묶음만으로는 다른 PC에서 실행되지 않는다. 실행은 원래 PC에서 `/home/binci/vlm_ws/route_console_20261122/run_console.sh`로 한다.

## 한계

- 좌표 변환 정확도를 재지 않았다(줄자 대조 없음).
- 그리퍼 크기와 도달 거리를 실측하지 않았다.
- 인식 결과에서 Nav2 주행으로 자동 전달하지 않는다. 경로는 파일로 전달한다.
- 장소에 놓기 규칙(팀원 정차 자리에서 구역을 향함)은 사용자 확인 전이다.
- 지시 문장의 다양성을 체계적으로 시험하지 않았다.
