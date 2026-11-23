# EXP110 채점·탐색 코드 독립 검토 (읽기 전용)

- 검토일: 2026-11-15 (KST)
- 범위: `tools/score.py`, `tools/dev_diag.py`, `tools/explain_examples.py`, `explore/X1_metrics/x1_metrics.py`, `explore/X4_block_zoom/x4_block_zoom.py`와 해당 `config.json`, 호출 경로인 `tools/run_models.py`
- 데이터 접근: P01–P04의 저장된 dev 결과와 P01 정답만 읽음. P05–P10 사진·결과는 읽지 않음. GPU 실행·원본 파일 수정 없음.
- 기준 코드 SHA256: `score.py` `16050d4fcac7de814ae3e5a27616996f6b7c79c2941aa3a71ff24afc1728f906`, `x1_metrics.py` `f69724cecf85b3b14922cdd7909b25c606b5f3af13b0d07a6c27c0c1268fb3e3`, `x4_block_zoom.py` `6bf188bc5b84d738870037b1e418e3ca2b320899f4cfd64a83faa62d4c3aee9b`, `dev_diag.py` `ec8cfed52799b95fa7842fe768545169c6f6863246c03b37e28042f7b5aa4e42`, `explain_examples.py` `2ba0f918b1096d1393f98e2d546e99161951bd644afbdae5e0b3253929fb1643`.

| 지적 | 심각도 |
|---|---|
| F1, F4 | 결과 수치가 바뀜 |
| F5, F6 | 결과 수치가 바뀔 수 있음 (현재 dev 영향은 미확인/없음) |
| F2, F3 | 개선 (test 봉인·필수 산출물) |
| F7, F8, F9, F10 | 표시만 틀림 또는 해석 제한 |

## 우선 수정할 지점

### F1. 부분 출력도 완전한 조건처럼 채점됨 — CONFIRMED, 결과 수치가 바뀜

- 위치: `tools/score.py:102–129`, `:155–178`, `:191–203`; X1도 `explore/X1_metrics/x1_metrics.py:75–96`에서 같은 방식으로 없는 사진을 건너뜀.
- 구체적 상황: 현재 `run02/C6-dino_s10k30`은 검토 시점에 P01–P03만 저장되어 있었다. `score.rows_for(..., 'dev', ...)`는 오류 없이 영어 147행·한국어 144행을 반환했다. 완전한 dev 조건은 영어 196행·한국어 192행이다. `summarize()`는 이를 정상 성공률처럼 출력하고 `paired()`도 존재하는 교집합만 비교한다.
- 확인 방법: 조건 폴더에서 P04 결과가 빠진 상태로 `rows_for()`의 `(cond,lang)` 행 수를 세면 위 값이 나온다. P04가 저장되더라도 누락 파일을 다시 만들면 같은 현상이 재현된다.
- 권고: 채점 전에 요청한 조건별로 P01–P04/질의 ID 97개의 완전성·유일성을 검증하고, 미완료 조건은 보고서에서 명시적으로 제외하거나 실패시킨다. test에서는 P05–P10에 같은 검사를 적용한다. 현재 score 결과를 조건 간 비교하기 전에 필수다.

### F2. `--split test --sweep`가 거부되기 전에 test CSV를 씀 — CONFIRMED, test 봉인 위험

- 위치: `tools/score.py:191–205`.
- 구체적 상황: `rows_for(..., 'test', ...)`가 test 정답과 예측을 읽고 `per_query.csv`까지 작성한 뒤에야 `a.split != 'dev'`를 검사한다. 이 명령은 종료 오류를 내지만 test 질의별 결과 파일이 남는다.
- 확인 방법: 코드 순서를 따라가면 된다. test를 실제 실행해 검증하지 않았다.
- 권고: `--sweep`와 `--split test` 조합 거부를 `rows_for()`와 출력 디렉터리 생성 앞에 둔다. test 실행 전 봉인 규칙이다.

### F3. 요청한 클래스별 성공률이 `summary.json`에 없음 — CONFIRMED, 표시·산출물 누락

- 위치: `tools/score.py:133–152`.
- 구체적 상황: `rows_for()`는 `class`를 각 행에 저장하지만 `summarize()`는 `(cond,lang)`과 `(cond,lang,type)`만 묶는다. 합성 입력 `class='drum'`으로 확인한 키도 `C6-dino|en`, `C6-dino|en|Q1`뿐이다.
- 확인 방법: `summary.json`의 `groups` 키에 클래스별 키가 있는지 확인한다.
- 권고: 클래스별 성공률이 아침 보고서의 필수 수치라면 동일한 행을 `(cond,lang,class)`로도 집계한다. 기존 주 지표 집계는 보존한다.

### F4. X1의 C6 ‘첫 박스’가 Qwen 출력 순서가 아님 — CONFIRMED, 탐색 지표 수치가 바뀔 수 있음

- 위치: `explore/X1_metrics/x1_metrics.py:59–68`, `tools/run_models.py:144–155`.
- 구체적 입력: Qwen 원문 `{"ids":[3,1]}`은 `parse_c6()`에서 `[1,3]`으로 정렬된다. X1은 저장된 `chosen_ids[0]`을 쓰므로 3번 대신 1번을 ‘첫 박스’로 평가한다. 1번 박스가 오답이고 3번 박스가 정답이면 TR은 실제 응답 순서 기준 성공인데 현재 구현은 실패다.
- 확인 방법: CPU에서 `run_models.parse_c6('{"ids":[3,1]}',{1,3})`의 결과가 `[1,3]`임을 확인했다.
- 권고: 과거 저장 결과의 `chosen_ids`만으로 원래 순서를 복원할 수 없다. `raw`에서 원래 `ids` 순서를 재파싱하거나, 지표 이름을 ‘선택된 박스 중 가장 낮은 후보 번호가 정답’으로 바꿔 해석한다. 후보 번호는 검출 점수순이므로 현재 지표는 사실상 선택 후보 중 최고 점수의 적중이다. C1은 저장 출력 순서, 검출기는 점수 순서가 코드 정의와 일치한다.

### F5. C6 `ok`는 일대일 TP와 다를 수 있음 — CONFIRMED 합성 사례, 현재 데이터 영향 UNKNOWN

- 위치: `tools/score.py:74–87` 대 `:40–48`, `:58–71`.
- 구체적 입력: 서로 겹치는 정답 `[0,0,10,10]`, `[1,0,11,10]` 두 개와 후보·선택 박스 `[0,0,10,10]` 하나를 주면 `c6_breakdown()`은 `ok=2`, Hungarian 채점은 `tp=1, fn=1`이다. 실패 분해가 개체별 ‘후보/선택과의 독립적 겹침’이기 때문이다.
- 확인 방법: 위 입력을 두 함수에 넣어 결과를 확인했다.
- 권고: 실패 분해의 `ok`를 TP로 부르지 않는다. 실제 정답 박스가 서로 겹칠 수 있다면 선택 단계도 일대일 매칭으로 분해할지 검토한다.

## 해석과 경계 조건

### F6. X1의 CU ignore 기준이 주 채점 기준과 다름 — CONFIRMED 합성 사례, 현재 dev 수치 영향 없음

- 위치: `explore/X1_metrics/x1_metrics.py:38–50`, `tools/score.py:63–71`.
- 구체적 입력: ignore `[0,0,100,100]`, 예측 `[49,49,51,51]`, 정답 없음. CU는 예측 중심이 ignore 안에 있어 성공, strict는 IoU가 0.5보다 작아 FP 1개·실패다.
- 확인 방법: 두 함수 CPU 호출에서 `True`와 `False`가 나왔다. 현재 확인한 P01–P04 dev GT에는 ignore 0개라 이번 dev 수치에는 영향이 없다.
- 권고: 의도적으로 완화한 규칙이면 보고서에 명시하고, 같은 ignore 의미를 원하면 주 채점의 IoU 기준을 재사용한다.

### F7. X4 확대 효과 비교 시 토큰 설정·질의 집합이 다름 — CONFIRMED, 표시만 틀릴 위험

- 위치: `explore/X4_block_zoom/x4_block_zoom.py:80–90`, `explore/X4_block_zoom/config.json`의 `modes.oracle.queries`와 `modes.tiles.queries`.
- 구체적 상황: X4는 현재 `config.json`의 C1 토큰 상한 1024를 읽는다. run01 C1은 512였다. run01 C1과 X4를 바로 비교하면 확대와 출력 길이 변경이 섞인다. oracle은 블록 8개 질의, tiles는 소형 팔레트 4개를 포함한 12개 질의라 전체 평균도 직접 비교할 수 없다.
- 확인 방법: 각 run의 snapshot과 X4 manifest의 `max_new_tokens`, 두 질의 ID 목록을 비교한다.
- 권고: 같은 1024토큰 C1을 기준선으로 쓰고, 두 확대 방식의 비교는 공통 블록 질의에서 한다. Oracle은 정답 소형 팔레트 위치를 사용하는 상한 실험으로 계속 분리한다.

### F8. `dev_diag.py`의 ‘오답 선택’과 채점 FP는 다른 수치 — CONFIRMED, 표시만 틀릴 위험

- 위치: `tools/dev_diag.py:68–82`, `tools/score.py:63–71`.
- 구체적 상황: 진단은 Q5를 제외하고 예측 박스마다 아무 정답과도 IoU 0.5 이상 겹치지 않으면 `chosen_wrong`에 센다. run01 영어 dev에서 이는 173/366이다. 주 채점 FP는 182개(양성 질의 174, Q5 8)다. 중복 예측·Q5·무효 출력 처리 때문에 두 값은 다르다.
- 확인 방법: 저장된 C6-dino P01–P04에 `score_one()`을 재계산해 FP 182를 확인했다.
- 권고: 173은 ‘양성 질의에서 어느 정답과도 겹치지 않은 선택 박스’로 표기하고 FP와 같은 의미로 쓰지 않는다.

### F9. `explain_examples.py`의 DINO 순위는 C6에 표시된 후보 순위가 아님 — CONFIRMED, 표시만 틀릴 위험

- 위치: `tools/explain_examples.py:42–48`, `tools/run_models.py:167–177`.
- 구체적 상황: 설명 스크립트는 C2 원출력을 점수순으로 정렬한 순위를 쓴다. C6은 점수·면적 필터와 NMS를 거친 뒤 번호를 다시 붙인다. P01 `small pallet` 정답 두 개의 원출력 순위 48/30은 NMS·면적 필터 뒤 44/27이었다. 둘 다 기존 상위 15 밖이라는 결론은 유지된다.
- 확인 방법: 동일 C2 결과에 `run_models.candidates(..., min_score=.01, top_k=100)`를 적용해 위 순위를 확인했다.
- 권고: 그림 설명에서 ‘DINO 원출력 순위’라고 명시한다.

### F10. C2 sweep은 이산 격자이며 전체 최적값은 아님 — CONFIRMED, 해석 주의

- 위치: `tools/score.py:203–211`.
- 구체적 상황: 0.5 다음 점이 0.6이다. 저장된 dev 출력에서 0.55는 영어 성공 21/196, 0.5는 20/196이지만 0.55는 현재 sweep 표에 없다. dev 결과를 보고 탐색 범위를 넓혔다면 이 수정은 탐색적 변경으로 기록해야 한다.
- 확인 방법: `rows_for(run01,'dev',{'C2':0.55},queries)`와 0.5를 비교한다.
- 권고: ‘사전 지정된 격자에서 최고’라고만 쓰고 연속 임계값의 최적값이라고 주장하지 않는다. test를 보기 전에 선택 규칙과 값을 고정한다.

## 확인된 정상 동작

- `tools/score.py --selftest`, `explore/X1_metrics/x1_metrics.py --selftest`, `explore/X4_block_zoom/x4_block_zoom.py --selftest`: 모두 `PASS` (qwen_vl 환경, CPU).
- 좌표: `run_models.parse_c1()`은 확대 crop 크기에서 정규화 0–1000을 픽셀로 변환하고, X4 `to_raw()`는 원본 crop 원점과 배율을 역적용한다. 합성 입력 crop `[100,200,260,360]`, 확대 배율 4.8, 정규화 전체 박스 `[0,0,1000,1000]`은 원본 `[100,200,260,360]`으로 복원됐다. 현재 oracle/tile 설정의 정수 픽셀 크기에서도 이 관계가 맞는다.
- 주 채점의 Hungarian 매칭은 중복 박스 하나를 FP로 센다. 정답 1개에 동일 예측 2개를 넣으면 `tp=1, fp=1, success=False`였다.
- X1은 `CFG['splits']['dev']`, X4는 `images_for('dev')`, `dev_diag.py`는 dev 목록, `explain_examples.py`는 P01을 사용한다. 이 경로들에서 test 이미지를 여는 코드는 확인되지 않았다. `score.py`는 명시적 `--split test`를 지원하며 F2의 순서 문제가 있다.

## 연구 설계 관점 (INFERRED)

- CU는 구역·관계의 맥락 이해가 아니라 ‘중심점과 개수를 이용한 완화된 위치 적중’이다. TR은 경로 도달 가능성이 아니라 ‘첫 선택 박스의 목표 적중’이다. 이름과 주장 범위를 맞춰야 논문에서 평가 지표가 과장되지 않는다.
- X4에서 crop별 결과를 합치는 NMS는 점수 없이 crop 순서의 첫 박스를 보존한다 (`x4_block_zoom.py:50–55`). 여러 crop의 예측이 겹치면 순서를 바꿀 때 최종 박스가 달라질 수 있다. 현 데이터의 영향은 아직 UNKNOWN이며, oracle과 tiles 결과를 해석할 때 합성 순서와 중복 제거 수를 기록하는 것이 좋다.

## 다음 연구를 위한 검증 가능한 가설 (INFERRED)

1. **후보 증가의 교환관계:** DINO 후보 포함률을 높여도 Qwen의 선택 오답이 함께 늘면 엄격 성공률은 개선되지 않는다. 같은 dev 질의에서 후보 포함률, 후보가 있을 때의 선택 적중률, FP, 지연시간을 설정별로 함께 기록하고 최종 test 설정은 사전에 하나로 고정한다.
2. **위치 사전 정보의 가치:** 블록 확대 oracle과 전역 타일의 차이는 ‘확대 효과’만이 아니라 소형 팔레트 위치 사전 정보의 효과를 포함한다. 같은 질의·토큰 상한·원본 좌표 채점으로 두 방식을 비교하면 그 비용을 분리해 볼 수 있다.
3. **구역 안전 지표의 독립성:** 물체 수준의 엄격 탐지 성공과 구역 침범 판정은 다른 문제다. 구역을 확정한 뒤에는 박스 중심점, 물체 footprint 교차, 경로 clearance 중 무엇이 실제 안전 조건인지 별도 지표로 정의해야 한다.
