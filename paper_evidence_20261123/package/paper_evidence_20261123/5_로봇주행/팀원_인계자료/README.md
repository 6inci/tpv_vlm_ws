# KRoC 2027 논문 실험 근거 인수인계 (Work 1차 통합본)

작성: 2026-11-21, 개발 PC Claude Code. 원고·원시 로그·저장소·Work 원본은 읽기만 했고, 이 폴더 밖은 바꾸지 않았다(젯슨 주소 기억 파일 갱신 가능성 1건 제외, 아래 진행 기록 16:24).
**Claude 채팅은 `CLAUDE_CHAT_HANDOFF.txt` 부터 읽는다.**

## 폴더 구성

| 경로 | 내용 |
|---|---|
| `CLAUDE_CHAT_HANDOFF.txt` | Claude 채팅용 시작 문서 (V2 원고 기준): 기준 원고, 쓸 수 있는 결과와 조건, 미확인 주장, 표·그림, 첨부 순서 |
| `v2_review/V2_원고_대조.md` | V2 원고(`논문초안_V2.hwpx`) 문장별 근거 상태와 고칠 방향 |
| `ch3_v2/` | 논문 3장 구성(3.1~3.5 + Table 1)에 맞춘 데이터 정리와 Table 1 초안 |
| `claim_to_experiment_matrix.md` | Work 주장 C01–C14 → 실험 X → 시행 X-R → 원시 경로 → 표·그림 → 검증 상태 |
| `experiment_inventory.csv` | 관련 실험 전체 42개 (실제 바닥·받침대·카메라·시뮬레이션·SIL·오프라인·소프트웨어 시험·제외 항목) |
| `minimum_additional_experiments.md` | 이번 논문에 꼭 필요한 추가 실험 4개 (목적·측정 항목·우선순위·이유) |
| `verification_report.md` | 자동 검증 결과, 기존 기록 대 재계산 차이, 계산 전제와 한계 |
| `tables/` | T-NAV-1·2·3, T-CTRL-1(+2), T-GEO-1 (md·csv) |
| `figures/` | F-NAV-1, F-PATH-1, F-CTRL-1, F-GEO-1 (png, 200 dpi) |
| `data/` | 재계산 CSV·JSON, 원시 자료 목록·해시, 시행 색인, 새 출처, 젯슨 조회 결과, 검증 결과 |
| `merged/` | Work JSON 병합본 (`claims_and_experiments_merged.json`, `sources_merged.json`) |
| `scripts/` | 분석 스크립트 s01–s09, `run_all.sh`, `common.py` |
| `archive/` | 이전 판 인계 문서 (S-P01 원고 기준) |
| `work_original/kroc_paper_handoff/` | 받은 Work 패키지 원본 (고치지 않음) |

## 재계산 방법

- 환경: 개발 PC Ubuntu 22.04, Python 3.10, numpy·matplotlib·PyYAML(시스템 패키지), ROS 2 Humble(`/opt/ros/humble`, s03 의 bag 파일 읽기에만, 스크립트가 스스로 불러옴). 네트워크·로봇·젯슨 불필요.
- 입력: `~/mecanum/mecanum_data` (reports·test_logs 제외), 저장소의 env2 지도 yaml·pgm, `work_original/`. 폴더가 날짜 폴더로 옮겨져도 이름으로 찾는다.
- 실행 (개발 PC 터미널, 이 폴더에서):

```bash
bash scripts/run_all.sh                 # s01~s07·s09 전체 재계산 (bag 해시 계산 포함, 수십 초~몇 분)
python3 -I scripts/s08_verify.py         # 다시 실행해 바이트 비교 + 파싱·ID·경로·숫자 검사 → data/verification_checks.json
```

| 스크립트 | 하는 일 | 주요 출력 |
|---|---|---|
| s01_manifest | PC 세션·bag·경로 기록·카메라 기록 목록, 크기·해시·기간 | data/pc_*.csv |
| s02_route_runs | 경로 주행 20회 진행률·도착 오차·횡오차·방향 틀어짐, summary.json 대조 | data/route_runs_recomputed.csv, route_summary_compare.csv, tables/T-NAV-1 |
| s03_bag_metrics | bag: 바퀴/명령 비·멈춤·DWB 주기·위치 토픽 주기, NavigateToPose 목표, 저속 시험 변위, 안전 정지 | data/bag_*.csv, safety_stop_events.csv |
| s04_camera_detect | 카메라 기록 검출률·놓침 (시행 구간별) | data/camera_detection*.csv |
| s05_inventory | 실험 목록·시행 색인·새 출처 | experiment_inventory.csv, data/run_index.csv, data/new_sources.json |
| s06_tables_figures | 표·그림, 기존 기록 대 재계산 | tables/, figures/, data/reported_vs_recomputed.csv |
| s07_merge | Work 주장·출처와 병합, 연결표 | merged/, claim_to_experiment_matrix.md |
| s09_ch3_table1 | 3장 Table 1 초안 (재계산 CSV 만 읽음) | ch3_v2/Table1_draft.md·csv |
| s08_verify | 검증 (3장 Table 1·V2 대조 문서 포함) | data/verification_checks.json |

## 진행 기록

| 시각 (KST) | 완료 | 다음 시작 위치 |
|---|---|---|
| 16:19 | 폴더 생성, Work ZIP(`kroc_paper_handoff.zip`)을 `work_original/`에 풀어 원본 보존 | 젯슨 조회 |
| 16:24 | 젯슨 읽기 전용 조회 → `data/jetson_readonly_listing_20261121.txt` (`메카접속` 경로, 주소가 바뀌었으면 `tools/pc_setup/.jetson_last_ip` 갱신됐을 수 있음) | PC 자료 목록 |
| 16:41 | s01 PC 자료 목록 (세션 21, bag 18, 경로 기록 20, 카메라 기록 13) | 경로 주행 재계산 |
| 16:51 | s02 경로 주행 20회 재계산, summary.json 160항목 대조 | bag 재계산 |
| 17:16 | s03 bag 재계산 1차 | — |
| (세션 끊김) | 17:18 이후 s03 수정 1줄·재실행이 승인 전에 끊김. 보고서·보관 폴더 조사 에이전트도 멈춤(그 뒤 직접 확인으로 대신함) | — |
| 재개 | s03 수정·재실행, s04 카메라 검출률, s02 진행률 투영 보완, s03 목표 종료 사건 보완 | — |
| **목록 완료** | s05 → 실험 42개, 시행 31개, 새 출처 32개 | 표·그림 |
| **주요 분석 완료** | s06 표 7개·그림 4개(눈으로 확인, 지도 위아래 뒤집힘·글자 겹침 고침), s07 Work 병합(주장 14, 실험 기록 42) | 추가 실험·검증 |
| 마무리 | 추가 실험 제안, s08 검증 35개 통과(재실행 바이트 동일), 검증 보고서, Claude 채팅 인계 문서. 원고 텍스트는 사용자 결정으로 넣지 않음 | 끝. 아래 '다음에 할 일' |
| 18시대 추가 | 논문 V2 3장 구성(3.1~3.5 + Table 1)에 맞춘 정리: `ch3_v2/3장_데이터정리.md`, `ch3_v2/Table1_draft.md·csv` (`scripts/s09_ch3_table1.py`). 2026-08 RMSE 1.17 cm 는 출처를 찾지 못해 제외(사용자 결정). zip(18:30)에는 들어 있지 않음 | V2 원고(HWPX)를 받으면 대조 |
| 23시대 추가 | V2 원고(`논문초안_V2.hwpx`, 18:43) 대조 → `v2_review/V2_원고_대조.md`. 인계 문서를 V2 기준으로 다시 씀(이전 판은 `archive/`). Table 1 의 소화기→보행자 통계를 도착 오차(SUCCEEDED 4회)와 끝 위치 오차(전체 6회)로 나눔. s08 에 3장·V2 검사 추가, `run_all.sh` 에 s09 추가 | — |
| 23시대 추가 2 | 사용자 맥락 메모 원문을 다시 읽고 빠진 것을 고침: 2026-08 수치는 세 개(RMSE 1.17·평균 1.05·최대 1.57 cm)이고 모두 출처 없음 → 세 값 모두 제외. 인계 문서 1절 (11)에 맥락 메모와 다르게 읽을 항목(검증 표현, 그리퍼 파지 테스트, 카메라 높이, 구역 이름, 사진 15장, 성공률 칸) 추가 | 끝. 사용자 결정 대기: 3.2·3.3 실험 여부, 원고 문장 수정 |

## Work 통합 상태와 다음에 할 일

- 이번 판 = GPT Work `kroc_paper_handoff.zip`(2026-11-21 16:13 받음)과의 **1차 통합본**. Work 의 주장·출처·계획·관찰 ID 와 문장은 그대로 두고 `claude_code_update`·`experiment_records` 칸만 더했다. 충돌한 내용(예: NavigateToPose 'ABORTED 3개', route2 '89 %·24 cm')은 양쪽 값과 근거를 `verification_report.md` 2절에 남겼다.
- Work 가 새 판을 보내면: 새 판을 `work_original_<날짜>/` 로 따로 풀고 `scripts/s07_merge.py` 의 입력 경로만 바꿔 다시 병합한다. 같은 분석은 `data/` 를 재사용한다.
- 원고가 다시 바뀌면: 새 판의 글자를 임시 폴더에서 뽑아 `v2_review/` 에 대조 문서를 하나 더 만든다(이전 대조 문서는 둔다). 원고 본문은 이 폴더에 넣지 않는다.
- 자료 전송(Claude 채팅 첨부, Work 회신)은 사용자가 한다. 첨부 순서는 `CLAUDE_CHAT_HANDOFF.txt` 5절.
- 선택: 젯슨에만 있는 bag 2개(11/20 21:18·23:46)를 PC 로 복사하면 s03 으로 X47·X48 도 계산할 수 있다.
