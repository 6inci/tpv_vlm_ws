# Autonomous Semantic Reasoning Probe (2026-11-18)

v0.2와 **독립**인 실험이다. Qwen3-VL-4B가 사진 한 장과 자연어 작업지시만 보고 사전 후보 없이 작업에 맞는 기능 영역을 스스로 제안하는지 본다.
결과와 해석은 [REPORT_2026-11-18.md](REPORT_2026-11-18.md), 배경은 [감사 보고서](../../docs/research_audits/2026-11-18_vlm_autonomy_audit.md).

## 지킨 경계

- 기존 v0.2 소스·설정·프롬프트·실행 결과·GUI·경로계획기·환경은 수정하지 않았다. 이 폴더 밖에서 바뀐 파일은 없다.
- 기존 모델 worker(`src/semantic_task_nav/perception/model_client.py`)를 import해서 그대로 쓴다. 모델 호출마다 **이미지 1장 + 프롬프트 1개**만 전달된다(테스트로 확인).
- 지도, 노란 경계, Z06, 사람이 그린 영역, 후보 영역, 개발 이미지용 질의 문구는 입력에 없다.
- 경로를 만들지 않고 하드웨어 명령을 보내지 않는다. 정확도는 계산하지 않는다(사람이 확정한 정답 없음).

## 파일

| 파일 | 역할 |
|---|---|
| `config.yaml` | 지시문, 조건, 이미지 레지스트리(해시, dev/eval 역할), 생성 설정 |
| `prompts.yaml` | 공통 머리말 + 조건별 사전 지식 블록(A/B/C1) + 출력 형식. 개정 이력 포함 |
| `prompts_frozen.json` | 평가 실행 전에 기록한 프롬프트 해시(사전 등록). 달라지면 eval 실행을 거부한다 |
| `probe.py` | 실행기: 프롬프트 생성, 모델 호출, 스키마 검증, 시각화, 매니페스트 |
| `compare.py` | 조건 간 비교표와 패널 이미지 (v3 형식 실행 전용) |
| `test_probe_offline.py` | GPU 없이 도는 검사 7건 |
| `data/images/` | 사용한 사진 3장의 복사본 (원본은 `~/Pictures/Calibration/`) |
| `results/<run_id>/` | 실행마다 새 폴더. 아래 참고 |

실행 폴더 구성: `manifest.json`, `<이미지>/<조건>/{prompt.txt, raw_response.txt, structured.json, result.json, overlay_areas.png}`, `comparison/{summary.md, summary.json, <이미지>_panel.png}`, `worker_logs/`.

## 실행

```bash
cd /home/binci/vlm_ws/semantic_task_navigation
PY=/home/binci/miniconda3/envs/stnav_gui/bin/python
$PY -m pytest -q experiments/autonomous_semantic_reasoning/test_probe_offline.py      # GPU 불필요
$PY experiments/autonomous_semantic_reasoning/probe.py prompts [--show]               # 해시·동결 상태
$PY experiments/autonomous_semantic_reasoning/probe.py run --split dev                # 개발 이미지
$PY experiments/autonomous_semantic_reasoning/probe.py run --split eval               # 동결 후에만 실행됨
$PY experiments/autonomous_semantic_reasoning/compare.py experiments/autonomous_semantic_reasoning/results/<run_id>
```

## 데이터 사용 규칙

- `dev` 이미지(D1)에서 출력 형식을 맞추는 동안 프롬프트를 3번 고쳤다(v0→v3, 개발 실행 4회, 이력은 `prompts.yaml`). 모두 공통 머리말과 출력 형식만 바꿨고 사전 지식 블록은 그대로다.
- v3를 동결한 뒤 `eval` 이미지(E1, E2)를 한 번씩 실행했다. 평가 이미지로 프롬프트를 바꾸지 않았다.
- 프롬프트를 바꾸려면 새 `prompt_version`과 새 동결 파일이 필요하고, 그 뒤의 평가는 새 이미지로 해야 한다.
- EXP110 P06(v0.2 개발 이미지, SHA-256 `d1984a19…`)은 실행기가 거부한다.
