# VLM 단일 이미지 실험 최종 보고서

## 실험 개요
단일 사시 상단 이미지에서 객체, 관계, 장면 맥락, 의미 구역, 이동 가능 영역을 추정하고 이미지 평면 지도로 시각화했다. 실제 로봇·ROS·Nav2는 실행하지 않았다.

## 사용한 입력 이미지
`/home/binci/vlm_ws/new_experiment_environment/approximate_environment.png` (1448×1086px, SHA-256 `02a4e79589db3c15529b372768ef349b4ef2e4ebe2efb2454f605be43eb95a95`)

## 사용한 환경 / 모델 / 코드 경로
- 모델: `Qwen/Qwen3-VL-4B-Instruct` (로컬 캐시, 결과 출처 `HYBRID_QWEN_OBJECTS_RELATIONS_MANUAL_REGIONS`)
- 실행 환경: `/home/binci/miniconda3/envs/qwen_vl`
- 코드: `/home/binci/vlm_ws/new_experiment_environment/scripts`
- 결과: `/home/binci/vlm_ws/new_experiment_environment/results/EXP001_run01`

## 진행 단계 요약
입력 확인 → 로컬 Qwen 추론 → 원문 보존 → JSON 검증 → 결정론적 좌표 변환 → 객체/구역/이동성 시각화 → 산출물 검증.

단계 판정: object recognition `PARTIAL`, object relation `PARTIAL`, scene context `PASS`, semantic region `PARTIAL`, traversability `PARTIAL`, semantic map `PASS`, integrated map `PASS`, natural-language grounding prototype `PASS`. `PARTIAL` 항목도 다음 단계 입력으로 사용할 수 있으나 사용자 검증 Ground Truth는 아니다.

## 결과 출처 구분
- 장면 맥락·객체 label·정규화 bbox: `MODEL_OUTPUT`
- 허용 predicate/유효 ID로 필터링한 관계와 pixel bbox 변환: `POSTPROCESSED` (원천은 `MODEL_OUTPUT`)
- 의미 구역·traversability polygon: `MANUAL_OR_AGENT_INFERENCE` preliminary annotation
- 자연어 명령→target region 연결: `HEURISTIC`
- 객체 overlay, semantic map, traversability map, 통합 지도: `POSTPROCESSED`

## 객체 인식 결과
Qwen 응답에서 완전히 닫힌 주요 객체 16개를 복구해 구조화했다. 좌표는 원본 이미지 기준 0..1000 정규화 bbox와 pixel bbox이며 metric 좌표가 아니다.

## 객체 관계 추론 결과
Qwen의 2차 응답에서 허용된 predicate를 사용한 공간 관계 5개를 복구했다. 이는 단일 이미지 기반 추론이며 물리 거리 측정이 아니다.

## 장면 맥락 이해 결과
An overhead view of a simulated warehouse environment with multiple robotic manipulators, mobile robots, and logistics equipment arranged on a black floor mat.

## 의미론적 구역 추정 결과
Qwen 구역 응답은 잘렸고 polygon도 퇴화되어 채택하지 않았다. 대신 수동 이미지 감사로 6개 구역을 객체 조합 근거와 함께 기록했다. 경계는 근사 이미지 평면 polygon이다.

## 이동 가능 영역 추정 결과
보이는 검은 바닥의 근사 open-floor polygon을 표시했다. 카메라 보정·homography·로봇 footprint/clearance가 없어 주행 계획에는 사용할 수 없다.

## 생성한 2D 지도
semantic, traversability, 두 정보를 합친 통합 지도를 생성했다. 모두 top-down metric occupancy map이 아닌 원본 이미지 overlay다.

## 자연어 명령과 연결 가능한 target region 예시
`natural_language_grounding.json`에 한국어 명령 4개와 target region ID를 연결했다. 명령 실행은 하지 않았다.

## 중간에 발생한 문제
- CUDA/NVML을 사용할 수 없어 Qwen을 CPU에서 실행했다.
- 최초 3000-token 추론 세션은 결과 없이 정지 상태가 되어 종료했다.
- JSON 파싱 상태: `primary response truncated/invalid: Expecting value: line 123 column 1 (char 3052); semantics response invalid: Invalid control character at: line 15 column 184 (char 1581)`.

## 시도한 해결 방법
드라이버나 환경을 변경하지 않고 생성 한도를 1200 token으로 줄여 동일 로컬 모델을 재실행했다. 파싱 실패 시에는 prototype 완성을 위해 수동 이미지 감사 fallback을 별도 출처로 표시한다.

## 해결되지 않은 문제와 영향
카메라 내부/외부 파라미터, 바닥 homography, 실측 scale, segmentation ground truth가 없다. 따라서 polygon·bbox는 의미 시각화용이며 안전한 metric 좌표·충돌 회피·경로 생성 근거가 아니다.

## 다음 실험에서 개선할 점
동일 장면의 보정된 카메라와 검증 기준점을 사용해 image-to-world 변환 오차를 평가하고, segmentation ground truth로 region/traversability IoU를 측정한다.

## 최종 산출물 감사
- 필수 결과: 12/12 확인. JSON은 모두 파싱되고 PNG 4개는 1448×1086으로 열렸다.
- 빈 파일: run 결과 디렉터리에서 0개.
- 중복 산출물: 없음. 두 raw 응답은 객체/맥락과 관계/구역을 위한 서로 다른 실제 추론 단계다.
- 실험 코드 실행 여부: `run_vlm.py`, `run_qwen_semantics.py`는 runtime JSON으로 실행을 확인했고, `build_outputs.py --self-check`도 PASS했다.
- 정리 필요 사항: Python 실행으로 생긴 `scripts/__pycache__/*.pyc` 3개는 연구 산출물이 아니므로 manifest에서 제외했다. 삭제 금지 원칙에 따라 파일 자체는 남겼다.
- 누락된 필수 결과: 없음.

### 완료 조건별 근거
1. Object recognition: `object_recognition.json` (16개 객체).
2. Object relation: `object_relations.json` (유효 관계 5개).
3. Scene context: `scene_context.json`.
4. Semantic region: `semantic_regions.json` (6개 preliminary region).
5. Traversability: `traversability.json` (이동 가능 영역과 제한 영역 7개).
6. Semantic 2D map: `maps/semantic_2d_map.png`.
7. Integrated map: `maps/semantic_traversability_2d_map.png`.
8. Natural-language grounding: `natural_language_grounding.json` (예시 4개).
9. 결과별 출처: 각 결과 JSON의 `source_type`과 `provenance.json`.
10. 실패·한계·해결·개선: 이 보고서의 해당 절.
11. 생성 파일과 용도: 아래 목록과 `FILES_MANIFEST.json`.
12. 최종 검증: JSON parse, 참조 ID, polygon 면적, PNG decode, manifest SHA-256 검사가 모두 PASS.

## 생성된 파일 목록
- `experiment_config.json`: 입력·모델·token 한도·run 출력 경로를 고정한 설정 snapshot이다.
- `scripts/run_vlm.py`: 기존 Qwen 유틸리티를 재사용해 첫 객체/맥락 추론을 실행한다.
- `scripts/run_qwen_semantics.py`: 관계·구역만 요청한 두 번째 축약 추론을 실행한다.
- `scripts/build_outputs.py`: raw 응답 복구·검증·좌표 변환·지도 생성과 자체 검사를 수행한다.
- `input_analysis.json`: 입력 경로·해상도·해시·시점 가정을 기록한다.
- `object_recognition.json`: Qwen 객체 label/bbox와 결정론적 pixel bbox를 보존한다.
- `object_relations.json`: 복구·필터링된 객체 관계 5개를 기록한다.
- `scene_context.json`: Qwen이 직접 생성한 장면 유형과 요약을 보존한다.
- `semantic_regions.json`: preliminary 의미 구역 6개와 근거 객체를 기록한다.
- `traversability.json`: qualitative 이동 가능 polygon과 제한/미확인 영역을 기록한다.
- `natural_language_grounding.json`: 한국어 명령 4개를 target region ID로 연결한 예시다.
- `provenance.json`: 결과별 출처 유형, 실패 원인, 사실·추론·미확인 항목을 통합 기록한다.
- `qwen_raw_output.txt`: 첫 Qwen 객체/맥락 원문 응답을 변경 없이 보존한다.
- `qwen_semantics_raw_output.txt`: 두 번째 Qwen 관계/구역 원문 응답을 변경 없이 보존한다.
- `vlm_prompt.txt`: 첫 추론의 재현용 프롬프트다.
- `vlm_semantics_prompt.txt`: 두 번째 추론의 재현용 프롬프트다.
- `runtime.json`: 첫 추론의 모델·환경·입력 해시·실행 시간을 기록한다.
- `runtime_semantics.json`: 두 번째 추론의 device와 실행 시간을 기록한다.
- `visualizations/object_recognition.png`: 16개 모델 bbox를 원본 이미지 위에 표시한다.
- `maps/semantic_2d_map.png`: 의미 구역을 표시한 image-coordinate semantic map이다.
- `maps/traversability_2d_map.png`: 이동 가능·제한·미확인 영역의 qualitative map이다.
- `maps/semantic_traversability_2d_map.png`: 의미 구역과 이동성 정보를 합친 image-plane map이다.
- `FINAL_REPORT.md`: 실험 결과, 실패, 한계, 해결 시도와 최종 감사를 설명한다.
- `FILES_MANIFEST.json`: 의미 있는 생성 파일의 크기·SHA-256·용도를 열거한다.
