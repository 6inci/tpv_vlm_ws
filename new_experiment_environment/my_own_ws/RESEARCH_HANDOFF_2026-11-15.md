# Codex 인수인계 — VLM 연구 상태와 다음 작업 (2026-11-15)

- 작성: Claude. 새 파일 하나이며 기존 코드·데이터·설정·문서는 수정하지 않았다.
- 표기: `CONFIRMED` 파일·결과로 확인 / `INFERRED` 근거 있는 해석 / `UNKNOWN` 미확인 / `PROPOSED` 제안(사용자 결정 전) / `DECIDED` 사용자가 결정
- 이 문서의 근거: `PROJECT_STATUS.md`, `EXPERIMENT_INDEX.md`, `REAL_ENV_EXPERIMENT_PLAN_v3.md`, EXP110–EXP120의 DESIGN·REPORT 문서, `EXP111_small_objects/CAPTURE_PLAN.md`, 이전 세션의 Claude Opus 연구 감사 보고서(2026-11-15, 작업 공간에 파일로 없음 — 아래 §4에 요약), 이번 세션의 읽기 전용 조사.
- 루트 규칙: `/home/binci/vlm_ws/AGENTS.md` (아래 §6). 작업 전에 반드시 읽을 것.

---

## 1. 최종 연구 목표와 담당 범위

### 1.1 목표 (DECIDED, 2026-11-15)

객체의 **상태·공간 관계·작업 맥락**을 이해해, 자연어로 지시된 **의미적 목적 영역**을 추론하고, **2D 지도에서 목표 지점과 장애물 회피 경로**를 생성한다. 고정 3인칭(외부) 카메라 기반, 미니어처 물류 환경.

### 1.2 개발 순서 (DECIDED)

| STEP | 지시 예 | 내용 | 핵심 능력 |
|---|---|---|---|
| 1 | "트럭 옆으로 이동해" | 객체 기반 목표·경로 생성 | 객체 위치 → 접근 목표점 → 경로 |
| 2 | "트럭 옆의 빈 팔레트로 이동해" | 상태·공간 관계 기반 대상 선택 | 빈/적재 상태 + 관계로 대상 하나 선택 |
| 3 | "하역장으로 이동해" | 작업 맥락 기반 의미적 목적 영역 추론 | 작업 의미 → 목적지 조건 → 영역 |

- 선행 STEP이 `PASS`가 아니면 다음 STEP을 구현하지 않는다 (AGENTS.md). 설계 문서화는 허용.
- 현재 위치: 실제로 측정된 것은 STEP 2의 구성요소 진단(EXP110–EXP120, dev 위주)이다. **STEP 1의 목표점·경로 코드는 `my_own_ws`에 없다** (`CONFIRMED`, 감사 시 grep: `findHomography`, A*, occupancy 없음). legacy에만 있다 (§3.3).

### 1.3 담당 범위 (DECIDED, 계획서 v3 §10)

- 사용자 담당: 자연어 명령 → 객체 탐지 → 목적 객체/영역 선택 → 목표 위치 생성 → 경로계획(경로 좌표와 시각화)까지.
- 팀원 담당: 실제 로봇 주행·제어, homography·calibration(E1), ArUco 로봇 위치. **실제 주행 검증은 팀원이 후속으로 한다.**
- 지금은 이미지(화면) 좌표로 끝까지 연결한다. homography는 사용자가 시작을 정할 때 E1 결과와 붙인다. 팀원의 homography·ArUco 산출물은 작업 공간에 없다 (`UNKNOWN`).
- MCP는 시간이 있을 때의 확장이다.

---

## 2. 실험별 진행 상태와 주요 결과

공통: 실제 카메라 데이터는 2026-11-14~15에 시작되었다. 9월의 합성·AI 생성 이미지 run은 실환경 증거로 쓰지 않는다 (계획서 v3 §0.1). 모든 수치 옆의 dev/test 구분에 주의.

### 2.1 요약표

| ID | 질문 | 상태 | 핵심 결과 | 위치 |
|---|---|---|---|---|
| EXP110 | 단어 → 박스. detector 후보 번호 + Qwen 선택(C6)이 Qwen 단독(C1)보다 나은가 | `PASS` (G1 = 사전 등록대로 실행됨). **가설 미확정** | test 영어 엄격 질의: C1 0.208, C6-dino 0.272 (+6.5 pp, 군집 bootstrap 95% CI [−0.3, +14.0] pp). 소형 팔레트·블록은 모든 조건 0 | `my_own_ws/EXP110_word_grounding/` |
| EXP110-X6 | 모델 크기: 4B bf16(M1) / 8B bf16 offload(M2) / 8B 4-bit(M3) | `FROZEN` (dev 탐색, test 미실행) | C1은 M1≈M3. C6에서 M3가 −7.6 pp (없는 물체에 번호 선택 8/16). 작업 물체는 모두 0/48. 8B는 병목 해결 못 함 | `EXP110_word_grounding/explore/X6_model_size/REPORT_2026-11-15.md` |
| EXP111 | 소형 팔레트·블록 찾기 | `DESIGN` (dev 탐색 완료, 새 촬영 전) | 외형 설명 + 확대 확인: 팔레트 8/8 (오탐 3), 상태 6/8, 색 4/4, 전 단계 정답 1/4 배치 | `my_own_ws/EXP111_small_objects/` |
| EXP112 | 후보 확대(H-zoom)가 상태 판단을 돕는가 | `IMPLEMENTING` (코드·selftest·smoke만, 정확도 결과 없음) | smoke: 형식 오류 Z-both 1건·B-rule 5건, GPU 최대 9.07 GiB | `my_own_ws/EXP112_state_zoom/` |
| EXP120 (= 계획서 v3의 E3a) | VLM 직접 관계·상태 판단(J-vlm)으로 팔레트 하나 선택 vs Python(J-geo) | `DESIGN` (dev 가능성 검증 완료, 정식 E3a 아님) | oracle: J-vlm 15/24 (잘못된 선택 4), J-geo 24/24(상한). detected: J-vlm 12/24 (잘못된 선택 6), J-geo 15/24 (잘못된 선택 1) | `my_own_ws/EXP120_E3a_relation/` |

### 2.2 EXP110 상세 (CONFIRMED)

- 데이터: P01–P10 (Cheese JPEG 1920×1080, 왜곡 보정 전). dev P01–P04, test P05–P10. 설정은 해시로 동결(`FREEZE_2026-11-15.sha256`), test는 조건당 1회 실행, 재채점 일치.
- 한계: 같은 장면·같은 고정물. 49개 질의 중 39–43개가 6장 모두 같은 결과 → 독립 표본이 아니다. McNemar p=0.0026은 참고만(사전 등록 근거는 군집 bootstrap).
- 병목: 후보 생성. test 정답 박스 744개 중 314개가 후보 밖. dev에서 후보 15→60개면 포함률 0.577→0.849지만 성공률 0.255→0.128.
- 관계 질의(X3, dev): 최고 13/48. 64건 중 29건에서 첫 박스가 기준 물체를 가리킴. 없는 기준("지게차 옆")에 16/16 박스를 냄.
- 확대(X4, dev): 위치를 알면 블록 색 5/8, 위치 없이 타일로 나누면 0/24.
- 보고서: `EXP110_word_grounding/results/run03/analysis_test_20261115T121130/TEST_REPORT.md`.
- **주의**: EXP110 test(P05–P10)는 EXP120 작업 중 정답 좌표가 터미널에 한 번 출력된 사고가 있었다 (`EXP120/REPORT_2026-11-15.md` 말미 기록). P05–P10은 이후 튜닝·평가에 재사용하지 않는다 (DECIDED).

### 2.3 EXP111 상세 (CONFIRMED, dev P01–P04만)

- 해상도: Qwen은 32×32 px가 토큰 하나. 블록 16–24 px = 0.5–0.8 토큰, 소형 팔레트 46–54 px = 1.4–1.7 토큰.
- 이름과 외형 불일치: 빈 소형 팔레트를 VLM이 "ventilation grille"로 읽음.
- 파이프라인: DINO 상위 100 → 크기 필터(35–75 px, **dev 정답 크기를 보고 정한 임시값**) → 후보 확대 + 외형 정의문 예/아니오(A안) → 팔레트 주변 확대로 블록 색 질의.
- 한계: dev 4장 모두 "적재 1 + 빈 1" 구성이라 색 4/4는 변별력이 낮다. 정의문·크기 필터·B-rule은 dev를 보고 정했으므로 새 데이터에서 재검증해야 한다.
- 새 촬영 계획: `EXP111_small_objects/CAPTURE_PLAN.md` — dev D01–D08, test T01–T16, 배치당 3프레임. 고정물은 옮기지 않고 소형 팔레트 2개와 블록 2개(빨강·파랑)만 옮김. 촬영 전 단계 (`DECIDED`: 환경 변경 보류, 소형 팔레트를 큰 상자로 바꾸는 안은 보류).

### 2.4 EXP112 상세

- 조건: Z-full / Z-crop / Z-both (+ 기준선 B-rule, B-const). 모델 M1 4B bf16 고정, oracle 후보만 실행(`DECIDED`: detected 후보는 추가 예외 미승인).
- 판정 기준: `EXP112_state_zoom/CRITERIA_REVIEW.md`(v3→v4 초안, 사용자 검토 전). dev 선별 규칙: k=3 순개선 등 (`config.json` screening_rule).
- 새 dev 데이터 폴더는 아직 없음. P01–P04는 형식·메모리 확인에만 사용, 정답 기반 튜닝 금지.
- 감사 지적: 확대 효과가 전혀 없어도 선별 기준을 통과할 확률이 약 0.29–0.35 (`analysis/k_screen_20261115T152710/SUMMARY.md`).

### 2.5 EXP120 / J-gate 상세 (CONFIRMED, dev 24항목, 정답은 초안)

- 질의 6개(e01 가장 가까운 / e02 옆 / e03 빈 / e04 블록 있는 / e05 파란 블록 / e06 없는 기준) × P01–P04. 정답 `referents_draft.json`은 Claude 초안이며 **사용자 검토 대기**, 특히 P01 e01.
- J-vlm 상태 판단 12개 중 6개(우연 수준), "빈 팔레트" 4장 모두 못 고름. 번호 상자 색 때문이 아님(D1 진단, 가설 기각).
- 없는 기준 물체(e06): J-vlm 4장 중 2장에서 팔레트를 고름. J-geo는 현장 목록으로 4/4 보류.
- J-gate: 사람이 쓴 목록(L) 기반 존재 확인은 오선택 2건을 막고 손실 0. detector만(D)은 아무것도 못 막음(DINO가 트럭을 forklift로 0.51–0.52). 일치 게이트(agree)는 오선택 6→0이지만 정상 선택 2건 차단. **현재 최종 시스템에 채택하지 않음, 변형·튜닝 금지 (DECIDED)**.
- 한계: 후보 2개(우연 50%), P01·P03·P04에서 "트럭에 가장 가까운" 팔레트 = "적재된" 팔레트, dev 지도는 정답 주석으로 만듦.
- "옆" 관계: 잠정 정의, 새 데이터에서 검증 (`RELATION_DEFINITIONS.md`). "가장 가까운"은 확정.

---

## 3. 현재 모델 구성과 핵심 파일

### 3.1 모델 (CONFIRMED)

| 역할 | 모델 | 환경 |
|---|---|---|
| VLM (기본, M1) | Qwen3-VL-4B-Instruct, bf16, greedy | conda `qwen_vl` |
| 후보 detector | Grounding DINO tiny (`IDEA-Research/grounding-dino-tiny`) | `qwen_vl` |
| 모델 크기 비교 (동결) | Qwen3-VL-8B-Instruct 가중치 캐시(rev `0c351dd…`), 4-bit용 env | `qwen_vl`, `qwen_vl_q4` (bitsandbytes 0.50.2) |
| dev 비교만 한 것 | YOLOE-26s(`~/weights/yoloe/`), YOLO-World s, Florence-2 base/large, SAM2.1 hiera tiny, CLIP ViT-B/32(`~/weights/clip/`) | `yolo_world`, `qwen_vl` |
| 하드웨어 | RTX 5070 Ti Laptop 12 GB | 화면 출력이 약 1.4 GB 사용 |

- 환경(`qwen_vl`, `yolo_world`, `qwen_vl_q4`, `env_isaaclab`)은 변경하지 않는다.
- 모델 교체 제안(PROPOSED, 근거: 이번 세션 웹 조사. 이 프로젝트 영상에서 검증된 바 없음): 조건 도출이 병목이면 Qwen3-VL-4B-Thinking, 탐지가 병목이면 LLMDet. 새 모델 설치는 사용자 승인 후에만.

### 3.2 핵심 코드·설정 경로

기준: `/home/binci/vlm_ws/new_experiment_environment/my_own_ws/`

| 용도 | 경로 |
|---|---|
| 카메라 촬영(3프레임, PNG) | `EXP110_word_grounding/tools/capture.py` |
| 정답 박스 주석 | `EXP110_word_grounding/tools/annotate.py` |
| EXP110 실행·채점 | `EXP110_word_grounding/tools/run_models.py`, `score.py` |
| EXP110 설정·질의·동결 | `EXP110_word_grounding/config.json`, `queries.json`, `FREEZE_2026-11-15.sha256` |
| 카메라 위치 확인 | `EXP111_small_objects/tools/camera_check.py` |
| EXP111 파이프라인 | `EXP111_small_objects/tools/{failure_analysis,s2_gated_pilot,s3_zoom_verify_pilot,s4_blocks_on_predicted,pipeline_eval}.py`, `config.json` |
| EXP112 | `EXP112_state_zoom/tools/{h_zoom,h_zoom_score,k_screen_check}.py`, `config.json` |
| EXP120 | `EXP120_E3a_relation/tools/{e3a_poc,jgate,referents_review}.py`, `config.json`, `referents_draft.json` |
| 객체 registry 초안(평가용) | `2026-11-14_object_registry_draft/object_registry_draft.json` |
| 구역 초안(사용자 확인 전) | `2026-11-15_zone_draft/semantic_zones_draft_v2.json`, `SEMANTIC_NOTES_v2.md` |
| 촬영 설정 | `config/capture_config.json` (device는 null, 외부 카메라 연결 후 지정) |

### 3.3 재사용 가능한 legacy 코드 (읽기 전용 참조)

경로: `/home/binci/vlm_ws/wanna_find_detailed_experimental_x,Y/src/vlm_reposition/`

- `planning/grid_astar.py`: `build_grid_map`, `astar_grid`, `plan_path`, `metric_to_grid`, `grid_to_metric`
- `planning/collision_geometry.py`: `build_collision_world`, `is_robot_center_valid`
- 한계: 합성 장애물에서만 검증(stress 12/12), 2016×1133 고정 geometry API는 수정 필요. 원본 수정 없이 필요한 부분만 `my_own_ws`로 이식하고 출처를 기록한다.
- 로봇 footprint 0.29 × 0.246 m (legacy `config/physical_geometry.json`, 계획서 v3 §0.3).

### 3.4 참고 문서

- 계획서: `new_experiment_environment/REAL_ENV_EXPERIMENT_PLAN_v3.md` (v2는 보존). 설계 노트: `my_own_ws/2026-11-15_E3_design_notes/NOTES.md`.
- 이번 세션의 미완성 설계 보고서(작업 맥락 이해 실험 설계): `my_own_ws/2026-11-15_task_context_design/RESEARCH_DESIGN_REPORT.md`. Part 3–9만 작성, 요약·Part 1·2·10·부록은 자리표시자. **PROPOSED 문서이며 확정 사항이 아니다.**

---

## 4. 확인된 문제와 미해결 사항

### 4.1 이전 감사(Claude Opus, 2026-11-15)의 지적 (CONFIRMED, 원문은 파일 없음)

1. 핵심 주장을 "VLM이 공간 관계를 직접 판단(J-vlm)"에 두면 자체 dev 결과·문헌과 어긋남. 관계 약점은 이미 알려져 있고 J-geo의 오선택이 더 적었음.
2. 지각 실패(소형 물체)가 추론 평가를 덮어, 관계·상태 "추론"을 따로 잴 수 없음.
3. VLM 필요성 미입증: 비-VLM 기준선(키워드 B0, HSV 색 규칙)이 없음.
4. 데이터: 장면 1개, 팔레트 후보 2개(우연 50%), dev 4장은 20회 넘는 탐색으로 소진.
5. 새 test(T01–T16)를 EXP111·EXP112·E3a가 순차 사용하면 오염 위험 → **통합 동결 후 test 1회** 권고.
6. 목표점·좌표 변환·경로가 미구현.
7. 감사의 권고: 핵심 질문을 "잘못된 목적지 비율을 검증·보류로 줄이는가"로 옮길 것. **사용자는 이 권고만으로는 "작업 의미 이해와 목적지 추론"이라는 핵심이 약해진다고 판단했다.** 이 사용자 판단을 존중한다(§5).

### 4.2 선행연구 확인 (이번 세션 웹 조사, 페이지를 열어 확인한 항목)

- "VLM/LLM이 지시문에 없는 목적지를 추론해 이동 목표로 삼는" 방식은 **이미 구현되어 있다**: Mobility VLA(암시적 지시, 실제 로봇), NavA³(실제 로봇, 상위 시점 렌더 + GPT-4o), ConceptGraphs(어포던스 질의), SayPlan, Demand-driven Navigation 계열.
- 이들은 모두 로봇 탑재 시점이거나 사전 구축한 지도·스캔을 쓴다. 고정 외부 카메라 + 논리 상태 조건 + 보류 + 짝 대조 평가의 조합은 찾지 못했다 (약 30개 질의 기준 "못 찾음"이지 "없음"이 아니다).
- Select2Plan(고정 외부 카메라)은 목표가 영상에 이미 표시되어 주어지므로 목적지 추론이 아니다.
- 따라서 "VLM이 목적지를 추론한다" 자체는 독창성 주장이 될 수 없다. 남는 후보: 고정 외부 카메라 설정, 작업 상태 조건의 짝 대조 평가 프로토콜(같은 지시, 상태만 다른 장면), 보류 평가, 오류 단계 분해.
- 평가 방법론 근거: 짝 대조 점수(NaturalBench, Winoground, contrast set), 영상 없는 기준선(OpenEQA, MMStar), 보류 점수(GRES N-acc/T-acc), 소표본 통계(Bowyer et al. 2025). 

### 4.3 미해결 (UNKNOWN / 사용자 결정 대기)

- EXP120 정답 초안 검토 (특히 P01 e01), 2차 주석자(팀원) 참여 범위.
- 새 촬영 미실시: dev 8 / test 16, 카메라 위치 확인, M00(고정물만 있는 장면).
- 구역 의미·접근 규칙 미확정: Z01 보행 통로를 로봇이 통과 가능한가, Z02 주행 범위, Z05 충전·Z06 하역장의 노란 선 통과 지점(`SEMANTIC_NOTES_v2.md` 질문 5개).
- 외부 카메라 연결 상태: `/dev/video0–3`이 노트북 내장 카메라였음 (2026-11-14 확인). 현재 연결은 `UNKNOWN`.
- 기존 calibration은 채팅 스크린샷 재구성이라 사용 불가, 팀원이 재calibration 예정(`config/capture_config.json`의 `calibration_status`).
- 트럭 적재함 상자를 손으로 치울 수 있는가, 충전 패드 위에 팔레트를 올려도 되는가 — 미확인. (작업 맥락 설계에 필요)
- 8B 재검토 조건("정확한 후보를 받아도 4B가 반복 실패") 충족 여부 미판정.
- `NEXT_STEP.md`, `DECISIONS.md`, `EXPERIMENTS.md`가 작업 공간에 없음. 상위 `PROJECT_STATUS.md`에 EXP111·112·120·X6 미반영. `vlm_ws`는 git 저장소가 아님.

---

## 5. 확정된 결정과 검토 중인 제안

### 5.1 확정 (DECIDED, 사용자)

- 연구 목표와 STEP 1→2→3 순서 (이 요청 문서).
- 실제 로봇 주행·제어는 팀원이 후속 검증.
- 계획서는 새 파일로 갱신하고 기존 자료는 보존 (v3).
- E3a(대상 선택)와 E3b(목표점, 결정론적 모듈)를 분리. E3b는 삭제하지 않음.
- 최종 시스템 구조는 **확정 보류**. M-map(고정물 = 사전 지도, 이동 물체 = detector)이 우선 구현 후보, M-det는 비교.
- 정보 3분류: 사전 지식(어휘, 로봇 치수, 고정 시설 위치) / 관측 정보(detector·Qwen 출력) / 평가 정답(채점기만 접근).
- 고정물은 옮기지 않고, 이동 물체(소형 팔레트·블록)의 위치·적재 상태로 정답을 바꿈. 관계 정답은 사람이 작성(Python 규칙으로 자동 생성 금지).
- 기본 모델 M1(Qwen3-VL-4B bf16). 8B 4-bit 채택 안 함, 양자화가 원인이라고 단정하지 않음.
- J-gate 탐색 종료, 최종 시스템 미채택, 변형·튜닝 금지.
- 환경 변경 보류(소형 팔레트 → 큰 상자 보류).
- EXP110 P05–P10은 튜닝에 재사용하지 않음. EXP120 detected 조건은 "dev 탐색" 한정 예외.
- EXP112: oracle 후보만 실행, 모델 M1 고정, 8B 미사용.
- 의미 구역 추론(③)은 E4-lite의 작은 보조 실험: 구역 경계(번호)는 사전 지식으로 주고 **이름은 주지 않는 것**이 주 조건.

### 5.2 검토 중 제안 (PROPOSED — 사용자 결정 전, 확정으로 취급하지 말 것)

- (감사) 통합 동결: 새 test를 EXP111·EXP112·E3a 전체에 한 번만 사용. B0 키워드 기준선, HSV 상태 규칙, 2차 주석자, test 정답을 작업 공간 밖에 보관.
- (이번 세션 설계) 핵심 가설을 "같은 지시, 작업 관련 상태만 바뀐 장면 쌍에서 목적지를 올바르게 바꾸는가"로. 구조: 조건 도출(VLM) → 후보(지도+detector) → 조건 검증(Python 관계 + VLM 국소 상태) → 선택·보류(Python) → 목표점 → A*.
- 비교군: TC-cond(제안) / TC-direct(VLM이 한 번에 선택) / TC-rule(키워드 + 사람이 쓴 규칙표) / TC-blind(영상 없음). 오라클 조건·상태는 진단.
- 보충 촬영: 기존 배치 직후 한 가지만 바꾼 배치(dev 4, test 7+선택 4), 트럭 적재함 상자·충전 패드 점유를 바꿈. 주 지표는 장면 쌍 정답률, 안전 지표는 잘못된 목적지 비율, 통계는 정확 부호 뒤집기 순열 검정.
- 상세는 `RESEARCH_DESIGN_REPORT.md` Part 3–9. 이 안은 STEP 3(작업 맥락)에 해당하며 STEP 1·2가 `PASS`하기 전에는 구현하지 않는다.

---

## 6. 기존 데이터·실험 환경 보존 원칙

루트 `AGENTS.md`(`/home/binci/vlm_ws/AGENTS.md`)를 따른다. 요약:

- **수정 금지**: 기존 코드, 실험 결과, 설정, 정답, 동결 해시, 계획서 v2·v3, 9월 합성 run, legacy 저장소. 새 run은 새 폴더(`results/EXPxxx_runNN/` 규약)에 만들고 기존 결과를 덮어쓰지 않는다. 실험 변형은 파일을 복사해 설명적인 이름으로 분리한다.
- 읽기 전용 참조 저장소: `/home/binci/camera_calibration_1920x1080`, `/home/binci/qwen_vision_test`, `/home/binci/mecanum_ros2`. 필요한 최소 코드만 이식하고 출처를 문서화.
- 환경(OS, 커널, 드라이버, CUDA, ROS2, conda, 패키지) 변경과 모델 설치·다운로드는 이유·영향·rollback을 설명하고 승인받은 뒤에만.
- dev/test 분리: test는 설정·정답 동결(sha256) 후 사용자 승인 하에 조건당 1회. 결과를 본 뒤 모델·프롬프트·평가 기준을 바꾸지 않는다(바꾸면 새 판, 사후 분석 표기). dev는 반복 탐색으로 소진되었으므로 새 dev가 필요하다.
- 재현성: 수치는 저장된 raw에서 재계산 가능해야 한다. VLM 원문 응답 보존, 스키마 검증, 결정론적 계산은 Python, 좌표계·단위·왜곡 여부 명시(`*_raw_px`, `*_undist_px`, `*_world_m`).
- 역할 경계: VLM은 라벨·박스만 출력하고 좌표·경로·충돌·모터 명령을 만들지 않는다. 불확실 시 기본 명령은 `cmd_vel = 0`.
- 상태값은 `NOT_STARTED, DESIGN, IMPLEMENTING, TESTING, PASS, FAIL, REWORK, FROZEN`만 사용. 사실은 CONFIRMED/INFERRED/UNKNOWN으로 구분.
- 사용자 지시 스타일(`new_experiment_environment/AGENTS.md`): 최소 diff, 변경 전 현재 상태·변경 내용·검증 방법 설명, 관련 없는 개선 제안 금지.

---

## 7. Codex가 이어서 수행할 작업과 우선순위

원칙: 아래는 **제안 순서**이다. 각 항목 착수 전에 사용자 승인을 받고, 실제 로봇 구동·모델 설치·촬영은 하지 않는다(사용자/팀원 몫).

| 순위 | 작업 | 선행 | 산출 | 비고 |
|---|---|---|---|---|
| 0 | 문서 정리: `NEXT_STEP.md`, `DECISIONS.md`, `EXPERIMENTS.md` 생성, `PROJECT_STATUS.md`에 EXP111·112·120·X6 상태 반영 | 사용자 승인 | 문서 | 기존 내용은 덮어쓰지 말고 추가 갱신 |
| 1 | **STEP 1 구현 (화면 좌표)**: 사전 지도(고정물 박스·종류)와 detector 결과 → "트럭 옆" 접근 목표점 계산(E3b) → 주행 가능 영역 점유 격자 → A* 경로 → 시각화 | M00 촬영 또는 임시로 dev 주석, 구역 Z02 확정 | `my_own_ws/`의 새 폴더 | legacy `grid_astar.py`를 이식. 목표점·경로 유효율을 `EXP`로 기록. 좌표는 raw 픽셀임을 명시, homography는 후속 |
| 2 | 사용자 결정 정리 요청: 구역 접근 규칙(§4.3), 통합 동결 여부, 촬영 설계(팔레트 수·test 배치·보충 배치), 2차 주석자 | – | 결정 기록 | STEP 2 이후의 선행 조건 |
| 3 | 비-VLM 기준선 구현 (selftest까지): B0 키워드 parser, HSV 상태 규칙 | 승인 | 코드 | 감사 권고, 새 모델 설치 불필요 |
| 4 | 새 촬영 후 EXP111·EXP112 dev 실행, EXP120 정답 검토 후 `--rescore` | 촬영, 정답 | dev 보고서 | `CAPTURE_PLAN.md` 순서(카메라 확인 → dev → 확인 → test → 즉시 봉인) |
| 5 | **STEP 2**: "트럭 옆의 빈 팔레트" 대상 선택을 새 dev에서 검증 → 통합 동결 → test 1회 | 3, 4, 동결 승인 | 보고서 | J-geo 기반이 dev에서 안전했음. J-vlm은 기준선 |
| 6 | **STEP 3**: 작업 맥락(예: "하역장") 실험. `RESEARCH_DESIGN_REPORT.md`의 짝 대조 설계를 사용자가 승인하면 적용 | STEP 1·2 `PASS` | – | 사전 지식 추측과 구분하려면 영상 근거가 필요(짝 대조) |
| 7 | 팀원 연동: homography(E1) 결과와 ArUco 인터페이스 협의, 경로 좌표를 m 단위로 변환 | 팀원 | 인터페이스 문서 | 실제 주행은 팀원 |

미확인이라 착수 전 확인할 것: 외부 카메라 연결, 새 촬영 일정, 팀원 homography 일정, 구역 접근 규칙, 정답 초안 검토 결과.

---

## 8. 한 줄 요약

실환경 실험(EXP110–120)은 "전체 화면 VLM 직접 판단은 소형 물체·관계·보류에서 불안정하고, 사전 지도 + Python 판정이 더 안전했다"는 dev 신호를 얻었으나 표본이 작고 정답이 초안이다. 목표점·경로는 미구현이므로, **STEP 1(객체 기반 목표·경로)부터 화면 좌표로 구현하고**, 상태·관계 대상 선택(STEP 2)과 작업 맥락 추론(STEP 3)은 새 dev·통합 동결 규칙에 따라 단계적으로 진행한다.
