# 다중 모델 작업 대상 추론 (multimodel_target_viewer)

사진을 고르고 자연어 작업지시를 입력하면 네 단계가 차례로 동작합니다.

| 단계 | 담당 | 하는 일 |
|---|---|---|
| 1 | **Grounding DINO** | 후보 위치를 찾습니다 |
| 2 | **SigLIP2** | 후보가 어떤 종류의 물체인지 비교합니다 |
| 3 | **VLM** | 작업지시의 뜻과, 어느 후보를 옮기고 어디에 놓을지를 판단합니다 |
| 4 | **Python** | 응답을 검증하고 작업장 정책을 적용합니다 |

결과는 원본 사진 위에 다음과 같이 표시하고, 한국어 작업 설명과 함께 보여 줍니다.

```
무엇을 찾았다 → 무엇을 집는다(A) → 어디로 간다 → 어디에 놓는다(B)
```

이 프로그램은 `instruction_target_viewer`와 **별개의 실험**입니다. 기존 프로그램의 파일은 import만 하고 바꾸지 않습니다.

## 실행

```bash
cd /home/binci/vlm_ws/semantic_task_navigation
/home/binci/miniconda3/envs/stnav_gui/bin/python experiments/multimodel_target_viewer/mm_app.py            # GUI
/home/binci/miniconda3/envs/stnav_gui/bin/python experiments/multimodel_target_viewer/mm_app.py --headless \
    --photo U25070552 --instruction "컨베이어 주변에 있는 빨간 큐브를 트럭 근처의 적절한 장소로 옮겨줘."      # 창 없이
```

- `--vlm qwen3vl_4b_bf16`을 주면 VLM으로 Qwen3-VL-4B를 씁니다. GUI에서는 "VLM" 목록에서 고릅니다.
- 다른 모델 프로그램이 GPU를 쓰고 있을 때는 실행하지 마세요.
- 사진을 처음 쓸 때는 DINO 탐지, SigLIP2 비교(후보 약 150개 × 2장), 모델 로딩 때문에 몇 분 걸립니다. 같은 사진의 다음 지시부터는 탐지 결과를 다시 씁니다.

기존 프로그램은 따로 실행합니다(아래 "사진 이름 변경" 참고).

```bash
/home/binci/miniconda3/envs/stnav_gui/bin/python experiments/instruction_target_viewer/viewer_app.py
```

## 모델과 환경

새로 설치한 패키지는 없습니다. 기존 Conda 환경은 그대로 쓰기만 합니다.

| 역할 | 모델 | 실행 위치 |
|---|---|---|
| 후보 탐지 | IDEA-Research/grounding-dino-tiny (기존) | v0.2 작업 프로세스, `qwen_vl` 환경 |
| 물체 종류 비교 | **google/siglip2-base-patch16-384 (이번에 새로 받음, 공유 HF 캐시, 1.5 GB)** | 새 작업 프로세스 `mm_worker.py`, `qwen_vl_q4` 환경, fp16 |
| VLM (기본) | Qwen/Qwen3-VL-8B-Instruct, 4-bit NF4 | `mm_worker.py`, `qwen_vl_q4` 환경. 캐시 스냅샷 `0c351dd…`(캐시에 `refs/main`이 없어 커밋 지정) |
| VLM (선택) | Qwen/Qwen3-VL-4B-Instruct, bf16 | v0.2 작업 프로세스 |
| GUI | PySide6, v0.2 위젯 재사용 | `stnav_gui` 환경 |

- **SigLIP2 크기**: 더 큰 `google/siglip2-so400m-patch14-384`(4.5 GB)가 보통 더 정확합니다. 이 네트워크(약 0.6 MB/s)에서는 받는 데 약 2시간이 걸려 base로 시작했습니다.
  - 받은 뒤 `config.yaml`의 `siglip.model_id`만 바꾸면 됩니다.
  - 받는 명령: `HF_HUB_DISABLE_XET=1 /home/binci/miniconda3/envs/qwen_vl_q4/bin/python -I -c "from huggingface_hub import snapshot_download; snapshot_download('google/siglip2-so400m-patch14-384', allow_patterns=['*.json','*.safetensors','tokenizer.model'])"`
- **유료 API**: 이 세션에는 API 키가 없어 쓰지 않았습니다. 쓰려면 모델명, 요금, 이미지 전송 여부를 먼저 정해 승인해야 합니다.
- **GPU 12 GB**: 큰 모델은 한 번에 하나만 올립니다. VLM을 호출하기 전에는 SigLIP2를 CPU로 내리고, 반대쪽 VLM은 내립니다.

## 한 번의 실행 순서

1. **VLM 해석**: 옮길 물체, 출발 기준 시설과 관계, 목적지 의미, 목적지 기준 시설과 관계를 뽑습니다.
   - 프롬프트는 task_context `task_context_v1`(+ R1 규칙)을 그대로 씁니다.
2. **Grounding DINO**: 사진을 처음 쓸 때 한 번 탐지합니다.
   - 대상: 기준 시설, 로봇, task_context 어휘, 빨간 큐브, 콘
   - 그 뒤에는 지시에 처음 나온 물체 이름만 추가로 탐지합니다.
   - 빨간 큐브는 저장된 색 기반 후보(candidate_retention_policy)도 함께 씁니다.
3. **SigLIP2 기준 시설 확인**: DINO가 제안한 컨베이어, 트럭, 로봇 상자가 정말 그 시설인지 확인합니다.
   - 기준: 해당 라벨이 SigLIP2 **1위**여야 확인된 것으로 봅니다. 2위 규칙에서는 P4의 저장 구역 박스가 컨베이어로 잘못 확인됐습니다. E1과 P4를 보고 정한 값입니다.
   - 기존 프로그램의 Qwen 크롭 확인을 이것으로 대체했습니다.
4. **탐색 영역**: task_context 관계 규칙(`on` = 내부, `next_to`/`near`/`from` = 주변)을 따릅니다. 기준 시설이 없으면 사진 전체가 영역입니다.
5. **SigLIP2 물체 종류 비교**: 영역 안 후보마다 두 장면을 config의 라벨 목록과 비교합니다.
   - 물체만 자른 이미지
   - 주변까지 넣은 이미지
   - 점수는 **이 라벨들 안에서의 상대 점수**이며 분류 확률이 아닙니다.
6. **VLM 역할 판단**: 번호 이미지와 SigLIP2 상위 3개 점수(참고용, 틀릴 수 있음으로 명시)를 함께 넣습니다.
   - 출발 쪽: 옮길 물체인지 판단합니다.
   - 목적지 쪽: 적재 상태, 놓을 수 있는 면과 여유 공간, 작업 목적을 판단합니다.
   - 프롬프트에 "테두리 색은 물체 색이 아니다"를 넣었습니다(F001 대응).
7. **Python 검증과 결정**: 아래 표의 순서로 진행합니다.

| 검증 | 실패 시 |
|---|---|
| 응답 번호가 1..n 안인지, 중복이 없는지 | `MAPPING_ERROR` → 응답 거부 |
| 모든 번호를 평가했는지 | `MAPPING_INCOMPLETE` → 출발 쪽은 거부, 목적지 쪽은 경고 |
| 후보 5개 이상에 똑같은 답(모양·역할·적재 상태)을 했는지 (F001 유형) | `DEGENERATE_RESPONSE` → 응답 거부 |
| A: VLM이 하나만 골랐는지 | 여럿 → `AMBIGUOUS` (서로 IoU 0.7 이상인 같은 물체 박스면 하나로 봄) |
| A: 지시의 물체 종류 라벨이 SigLIP2 상위 2위 안에 있는지 ("물건" 같은 일반 명사는 생략. 2위는 E1 스모크 테스트를 보고 정한 값이라 검증되지 않음) | 없음 → `TYPE_UNCERTAIN` (A 미확정) |
| B: 바닥 면적의 50 % 이상이 보행자 통로(Z01)인지 | 후보 제외 (`POLICY_NO_PLACEMENT`, 사용자 규칙 "보행 동선은 물품 배치에 쓰지 않음") |
| B: 주행 구역에서 5 cm 이내이고 로봇 출발점에서 접근 자세에 도달하는지 | 기존 접근성 규칙. 모두 불가 → `NOT_ACCESSIBLE` |
| B: task_context Tier1/Tier2 결정 | 동률 → `AMBIGUOUS`, 적합 후보 없음 → `NOT_FOUND` |
| 선택된 ID가 탐지 후보에 있고 박스가 같은지 | `selection_check` 실패 |

모든 지시에서 A·B를 강제로 정하지 않습니다. A만 정해지면 A만 표시하고 B는 미확정으로 남깁니다.

## 화면

| 위치 | 내용 |
|---|---|
| 왼쪽 | 사진, 작업지시, VLM 선택, 규칙(R1/R0), 실행·중지, 진행 상황, GPU 사용량 |
| 가운데 | 결과 사진 |
| 작업 설명·결과 탭 | 한국어 작업 설명(찾은 것 → 집을 것 → 놓을 곳 → 계획), A·B 요약, 이 실행과 사진 준비 때 실제로 호출된 모델 |
| 모델별 단계 탭 | 단계, 상태(색), 그 단계에서 쓴 모델, 내용. 실패한 첫 단계가 표시됩니다 |
| 후보 판단 탭 | 후보별 SigLIP2 상위 3개, VLM 판단, 적재 상태, 접근 가능 여부, 배치 금지 구역 비율, 모델에 실제로 들어간 이미지 |
| 실행 기록 | 이 세션의 모든 실행. Ctrl+클릭으로 두 줄을 고르면 위아래로 비교합니다 |

결과 사진의 표시:

| 표시 | 뜻 |
|---|---|
| 빨간 실선 | A |
| 파란 실선 | B (구역이면 반투명 칠) |
| 초록 화살표 | 작업 방향 A→B. 주행 경로가 아닙니다 |
| `1 PICK`, `2 MOVE 3 PLACE` | 작업 단계 |
| × | B 근처 로봇 정차 위치 |
| 점선 | 탐색 영역, 미확정 후보(`A?`, `B?`), 접근 불가 또는 배치 금지로 빠진 후보 |

## 사진 이름 변경 (2026-11-19 21:54 확인)

`/home/binci/Pictures/Calibration/`의 `undistorted_*.png` 8장이 `1calibration.png`–`8calibration.png`로 바뀌었습니다. 내용은 기록된 세션 사진과 sha256까지 같습니다.

| 새 이름 | 세션 ID | 새 이름 | 세션 ID |
|---|---|---|---|
| 1calibration.png | U23678143 | 5calibration.png | **U25070552 (E1)** |
| 2calibration.png | U24768580 | 6calibration.png | U25137685 |
| 3calibration.png | U24837448 | 7calibration.png | U25967031 |
| 4calibration.png | U24898553 (P4) | 8calibration.png | U26007129 (D1) |

- **이 프로그램**은 사진을 파일 이름이 아니라 sha256으로 식별합니다. 이름이 바뀌어도 세션 ID, 저장된 큐브 후보, E1 바닥 변환을 그대로 찾습니다.
- **기존 `instruction_target_viewer`와 이전 실험 설정**은 E1 경로를 `undistorted_25070552942982.png`로 갖고 있습니다. 그래서 지금은 바닥 변환 단계에서 실패합니다(확인함).
  - 기존 파일과 사진 폴더는 바꾸지 않았습니다.
  - 기존 프로그램을 다시 쓰려면 원래 이름을 되살리면 됩니다. 사진 원본을 바꾸지 않는 방법은 같은 폴더에 원래 이름의 심볼릭 링크를 두는 것입니다(사용자가 결정).

## 저장

프로그램을 켤 때마다 `results/<날짜_시각>_session/`이 새로 생깁니다.

```
session.json                    실행 목록, v0.2 코드 해시 비교
photos/<ID>.json                사진별 DINO 후보, 기준 시설 확인(SigLIP2 상위 3), 모델 호출 기록
photos/<ID>_siglip/*.png        SigLIP2에 넣은 crop (물체만 / 주변 포함)
runs/run_<번호>_<ID>/
  run.json                      해석, 영역, 후보, SigLIP2, VLM 판단, Python 검증, A·B, 한국어 설명, 호출된 모델, handoff
  overlay.png                   결과 사진
  overlay_candidates.png        후보까지 표시한 사진
  inputs/*.png                  VLM에 넣은 번호 이미지
  qwen/*_prompt.txt, *.txt      VLM 프롬프트 원문과 응답 원문
  model_calls.json              모든 모델 호출 (모델, 커밋, 양자화, 해시, 토큰, 시간)
worker_logs/
```

## 점검

```bash
/home/binci/miniconda3/envs/stnav_gui/bin/python experiments/multimodel_target_viewer/test_mm_offline.py
```

모델 없이 다음을 확인합니다.
- 순수 함수
- 응답 검증. 보존된 F001 응답이 `DEGENERATE_RESPONSE`로 거부되는지 포함
- 정답 참조를 쓰지 않는지
- **기존 프로그램 파일이 작업 전 해시(`baseline_check/`)와 같은지**
- 한국어 설명과 그림

## 한계

- SigLIP2 점수는 라벨 목록에 따라 달라지는 상대 비교입니다. 작은 물체(큐브 약 15 px)는 crop을 크게 키우므로 신뢰도가 낮을 수 있습니다.
- 바닥 좌표와 접근성은 E1 임시 변환(RMS 3.57 cm)과 DRAFT 지도에 의존합니다.
- 그리퍼 도달 거리와 배치면의 물리적 크기는 확인하지 않습니다.
- 로봇 출발점은 확인된 로봇 박스의 중심입니다(task_context의 E1 외 사진 규칙).
- 이전 EXP110-X6 결과: 8B 4-bit는 번호 고르기 형식에서 과다 선택 경향이 있었습니다. VLM 목록에서 4B로 바꿔 비교할 수 있습니다.
