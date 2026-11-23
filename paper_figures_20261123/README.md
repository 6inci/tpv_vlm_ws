# KRoC 논문 그림 3개 (2026-11-23)

저장된 실행 기록만 읽어서 그렸다. 모델과 경로 계산은 다시 돌리지 않았고, 그림에 적힌 숫자는 모두 기록 JSON에서 읽은 값이다.

- 다시 만들기: `python3 -I -B make_figures.py`
- 출력: `out/*.png` (600 dpi), `out/*.pdf`
  - PDF는 Type 3 글꼴이다. 한글 글꼴(Noto Sans CJK)이 TTC 형식이라 Type 42로 넣을 수 없었다.

## Fig. 1 시스템 구성 (`fig1_system`) → 2.3절

**캡션 초안**

> 제안 시스템의 처리 흐름. Qwen3-VL-4B는 자연어 지시만 입력받아 작업 항목을 추출하며, 영상의 후보 탐지와 비교는 Grounding DINO-tiny와 SigLIP2가 담당한다. 대상 결정, 좌표 변환, 정차 목표 및 경로 생성은 규칙과 기하 계산으로 수행한다. 실제 로봇 주행(점선)은 생성한 경로 파일을 전달하여 수행한 별도 실험이다.

**코드와 대조한 결과**
- Qwen에는 영상을 넣지 않는다 (`hybrid_pipeline.py`, `nl_to_path.py` 2–9행).
- 시설은 DINO가 후보를 내고 SigLIP2 base가 글자 라벨("a truck")로 확인한다.
- "기준 시설 주변 영역" 판단: 시설 박스를 max(0.6×긴 변, 120 px)만큼 넓힌 영역 안에 큐브가 있는지 Python이 검사한다.
  - 기준은 박스 중심이 영역 안에 있거나, 박스의 50% 이상이 영역 안에 있는 것이다.
  - 근거 코드: `make_roi`, `in_region`
- 목적지의 공간 관계는 그림에 넣지 않았다. 구현이 다르고, 결과도 2건 중 정답 0이다.

## Fig. 2 공간 관계에 따른 선택·보류 (`fig2_relation`) → 3.1절

- 근거: `hybrid_exemplar_20261121/session_20261121_073316`
  - (a) `run_008` (사진 4)
  - (b) `run_002` (사진 1)
- 지시: "하역장 옆에 있는 물체를 컨베이어 벨트 옆에 놓아줘."
- 사진: 개발 사진 8장 (2026-11-19)
- 2/2, 6/6은 `task_evaluation.json`에서 센 값이다.
  - 보류 6건은 모두 `NOT_AT_REFERENCE`다. 즉 큐브는 찾았지만 하역장 옆이 아니었다.

**캡션 초안**

> 공간 관계 조건('하역장 옆')에 따른 대상 선택 결과. 주황 실선은 기준 시설(트럭), 점선은 그 주변 영역, 흰 상자는 큐브 예시 유사도 상위 후보이다. (a) 1위 큐브가 영역 안에 있어 선택한 경우 (2/2), (b) 1위 큐브가 영역 밖에 있어 선택을 보류한 경우 (6/6).

**본문에 같이 적을 것**
- 목적지 조건 "컨베이어 옆"은 2건 중 정답 0이다 (동률 1, 엉뚱한 곳 확정 1).
- 개발 사진 8장이고 기준값도 같은 사진에서 정했다.
- 정답은 AI 판독이다 (`AI_READ_NOT_HUMAN_CONFIRMED`).

## Fig. 3 대상 → 정차 목표 → 경로 (`fig3_goal_path`) → 3.2절

- 근거: `nl_to_route_20261122/results/run_20261122_all15_final/20261120_220632_view/nl_to_path.json` (논문 수치를 낸 기존 시스템의 15장 실행)
- 지시: "빨간 큐브를 비어 있는 팔레트로 옮겨줘."
- 지도: 팀원 env2_20261119 (1 cm 칸)
  - 밝은 부분 = 주행 영역(노란 선 안)
  - 차체 29×24.6 cm, 그리퍼 끝 19.5 cm는 팀원 `robot_params.json` 값이다. 그리퍼 끝은 추정값이다.
- (c)에서 경로 2는 겹치는 구간을 구분하려고 x·y로 2.5 cm 띄워 그렸다.

**캡션 초안**

> 대상 선택에서 주행 목표와 경로 생성까지의 예. (a) 영상에서 선택한 큐브와 빈 팔레트, (b) 지도 위 물체 위치(×)와 정차 목표(사각형: 차체, 굵은 선: 그리퍼 끝), (c) 주행 영역 안에서 생성한 경로.

**대표 예로 고른 이유와 본문에 같이 적을 것**
- 15장 중 9장에서 두 구간 경로가 모두 나왔다 (`PATH_OK`).
  - 9장 모두 경로 전체의 차체 여유가 최소 3.0 cm였다.
  - 팀원 여유 검사에서 불가 0 cm, 빠듯함 0 cm였다.
- 9장 중 8장은 두 정차 위치 중 한 곳의 옆 어긋남이 6.3–7.8 cm다. 기준 2 cm를 넘지만, 이 기준은 표시만 하고 경로 생성을 막지 않는다.
  - 두 정차 위치가 모두 기준 안에 든 사진은 220632 하나뿐이어서 이 사진을 골랐다.
  - 본문에 "대표 예"라고 밝히고, 옆 어긋남 결과를 한 문장으로 적어야 한다.
- 그리퍼 끝과 물체 사이의 남은 거리는 0.3–8.4 cm다 (9장 기준). 그리퍼는 실측하지 않았다.

---

# v2: 영어판, PPT 도식 스타일 (`out_v2/`)

- 만들기: `python3 -I -B make_figures_v2.py`
- 근거 기록과 숫자는 v1 과 같다. 기록을 읽는 함수는 `make_figures.py` 의 것을 그대로 가져다 쓴다.
- 출력: `.png` (600 dpi), `.pdf`, `.svg`. 글꼴은 Liberation Sans(Arial 과 같은 폭)이고, ✓ ✕ 아이콘만 DejaVu Sans 다.
- 색 (dataviz 기본 팔레트)
  - 주황 #eb6834 = 집을 물체 / 경로 1
  - 파랑 #2a78d6 = 놓을 곳 / 경로 2
  - 상태색(초록 = 선택, 빨강 = 보류)은 아이콘과 글자와 함께만 쓴다.
- 영어 지시문은 실제 한국어 지시를 옮긴 것이다.
  - Fig. 1: "빨간 큐브를 비어 있는 팔레트로 옮겨줘."
  - Fig. 2: "하역장 옆에 있는 물체를 컨베이어 벨트 옆에 놓아줘."
- Fig. 2 는 보기 쉽도록 영역 밖을 어둡게 했다. 확대 창은 원본 화소 그대로다.

## 영어 캡션 초안

- **Fig. 1.** Overview of the proposed pipeline. Qwen3-VL-4B-Instruct parses only the instruction text into fixed task slots; candidate detection and matching on the camera image are performed by Grounding DINO-tiny and SigLIP2. Target selection, pixel-to-map conversion, stop-pose generation and path planning are rule-based geometric computations. Robot driving (dashed) was a separate experiment using the exported route files.
- **Fig. 2.** Target selection under a spatial-relation condition ("next to the loading dock", development images). The dashed box is the vicinity of the landmark (truck) verified by SigLIP2 base, and white dashed boxes are other cube candidates; areas outside the region are darkened for visibility. (a) The top-1 cube lies inside the region and is selected (2 of 2 images). (b) The top-1 cube lies outside the region and the selection is held (6 of 6 images).
- **Fig. 3.** From target selection to stop pose and route (representative image). (a) Selected cube and empty pallet with exemplar similarity. (b) Object positions (×) and stop poses on the map; rectangles are the robot footprint (29 × 24.6 cm) and bars the estimated gripper tip. (c) Routes planned within the drivable area; route 2 is drawn with a 2.5 cm offset where the two routes overlap.

---

## 밝은 색 실험 자료: 표·그래프·경로 시각화 각 1개 (`out_bright/`)

흰 배경에 하늘색·민트·살구색을 사용한 별도 자료다. 기존 `out/`, `out_v2/`와 제작 코드는 보존했다.

- 다시 만들기: 이 폴더에서 `python3 -I -B make_experiment_bright.py`
- PNG는 600 dpi, PDF와 SVG도 함께 저장한다. 한글 PDF는 기존 v1과 같은 Type 3 글꼴이고, SVG의 글자는 경로로 저장한다.
- JSON의 집계 수치, 사진 SHA-256, 경로 수와 여유 조건, 텍스트의 출력 경계 포함 여부를 생성 시 검사한다.
- 수치의 원본 경로와 재집계 결과: [`source_metrics.json`](out_bright/source_metrics.json)

| 자료 | PNG | PDF | SVG | 편집용 표 |
|---|---|---|---|---|
| 01. 다섯 후보 선택 방식 비교표 | [PNG](out_bright/01_method_comparison_table.png) | [PDF](out_bright/01_method_comparison_table.pdf) | [SVG](out_bright/01_method_comparison_table.svg) | [CSV](out_bright/01_method_comparison_table.csv) |
| 02. 공간 관계 결과 그래프 | [PNG](out_bright/02_spatial_relation_chart.png) | [PDF](out_bright/02_spatial_relation_chart.pdf) | [SVG](out_bright/02_spatial_relation_chart.svg) | — |
| 03. 대상→정차 목표→경로 | [PNG](out_bright/03_goal_path_visualization.png) | [PDF](out_bright/03_goal_path_visualization.pdf) | [SVG](out_bright/03_goal_path_visualization.svg) | — |

**01 근거와 캡션 초안**

- 원본: `hybrid_exemplar_20261121/ablation_20261122_140934/ablation_summary.json`
- I1 분모 8은 각 방식의 저장된 상태 목록 길이로 확인한다. I2 수행 가능 분모 2는 실제 실행의 `task_evaluation.json` 집계와 대조한다.
- Qwen의 후보별 이름 선택은 저장된 측정 결과다. 이 제거 비교의 지시 항목은 올바르게 해석됐다고 가정한 재생 조건이며, 별도 실제 GPU 지시 해석에서도 16/16 같은 답이었다는 원본 README의 조건을 함께 유지한다.

> 동일한 개발 사진 8장과 후보 crop에서 후보 선택 방식 다섯 가지를 비교한 결과. I1은 큐브와 빈 팔레트를 모두 올바르게 선택한 경우, I2는 수행 가능한 사진 2장에서 ‘하역장 옆’ 대상을 올바르게 선택한 경우이다. 선택 성공은 실제 주행 또는 물체 이송 완료를 뜻하지 않는다.

**02 근거와 캡션 초안**

- 원본: `hybrid_exemplar_20261121/session_20261121_073316/task_evaluation.json` 및 보류 6건의 `run.json`.
- 대상 선택 정답 2/2, 올바른 보류 6/6, 목적지 정답 0/2를 각각 다른 분모로 표시했다. 보류 이유는 모두 `NOT_AT_REFERENCE`다.

> 공간 관계 지시의 출발지 대상 선택 및 목적지 판정 결과. 출발지 관계 조건을 만족한 2장에서는 대상을 올바르게 선택했고, 만족하지 않은 6장에서는 선택을 올바르게 보류했다. 대상을 선택한 2장의 목적지 관계 정답은 0건이었다. 정답은 AI 판독이며 사람이 확인하지 않았다.

**03 근거와 캡션 초안**

- 원본: `nl_to_route_20261122/results/run_20261122_all15_final/*/nl_to_path.json` 15개. 대표 사진은 기존 Fig. 3과 같은 220632다.
- 두 구간 경로 및 팀원 여유 검사 통과 9/15, 전체 경로 최소 차체 여유 3.0 cm, 불가·빠듯함 0을 기록에서 재확인했다.
- 두 정차 위치 모두 옆 오차 기준 2 cm 이내인 사진은 1/9이며 나머지 8장은 한 곳이 6.3–7.8 cm 어긋난다. 이 제한은 그림 안에도 명시했다.
- 이 버전은 경로 좌표를 이동하지 않고 실선과 점선으로 겹침을 구분한다. 기존 Fig. 3의 2.5 cm 표시 오프셋을 적용하지 않았다.

> 대상 선택, 지도 위 물체 위치와 정차 목표, 두 구간 경로를 연결한 대표 예. ×는 물체 위치, 원은 정차 중심, 사각형은 차체, 굵은 선은 추정 그리퍼 끝이다. 두 정차 위치의 옆 오차가 모두 기준 이내인 유일한 사례를 선택했으며, 전체 경로 생성 사례 9장 중 나머지 8장에는 한 정차 위치의 옆 어긋남이 남아 있다. 실제 파지와 이송 성공은 검증하지 않았다.

원래 본문의 2.1절 예문은 이 자료 제작 범위에서 수정하지 않았다. 목적지 공간 관계가 구현돼 있다는 주장으로 사용하면 안 된다.
