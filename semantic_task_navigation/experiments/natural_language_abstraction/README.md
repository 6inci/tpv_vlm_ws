# Natural Language Abstraction Probe (2026-11-18)

지시문이 목적지 정보를 덜 줄 때(L1→L2→L3) Qwen3-VL이 작업환경의 의미를 얼마나 스스로 추론하는지, 언제 판단을 보류하는지 본 **탐색적 진단 실험**이다. 결과: [REPORT_2026-11-18.md](REPORT_2026-11-18.md)

- 분류: `EXPLORATORY_DIAGNOSTIC`. E1, E2는 [autonomous_semantic_reasoning](../autonomous_semantic_reasoning/REPORT_2026-11-18.md)에서 이미 쓴 사진이다. 독립적인 최종 성능 평가가 아니며, 정답 다각형이 없어서 정확도·IoU 정확도·성공률은 계산하지 않는다.
- 지시문 3개(사용자 문장 그대로): L1 "빨간 큐브를 트럭이 있는 하역장으로 옮겨줘." / L2 "빨간 큐브를 출하할 수 있는 적절한 장소로 옮겨줘." / L3 "생산된 빨간 큐브를 다음 물류 단계로 전달해 줘."
- 사전 지식 조건은 부모 실험의 A(없음)만 쓴다. 이미지, 모델(Qwen3-VL-4B bf16 greedy), 프롬프트 본문, 최대 토큰(2800)은 세 지시문에서 같다.

## 경계

- 부모 실험(`../autonomous_semantic_reasoning/`)의 `probe.py`와 `compare.py`는 **수정하지 않고** 파일 경로로 import해서 재사용한다(스키마 검증, 시각화, 비교 함수, 기존 모델 worker). 그 파일 해시는 실행마다 `manifest.json`의 `parent_files_sha256`에 기록된다.
- 기존 v0.2 소스, 기존 실행 결과, 부모 실험의 동결 프롬프트는 건드리지 않는다. 경로계획, ROS2, 하드웨어, 새 모델 설치는 없다.
- 모델에는 **이미지 1장 + 프롬프트 1개**만 전달된다(테스트로 확인). 프롬프트 머리말·출력 형식에는 트럭, 하역/출하 구역, 이 장면의 색·좌표·구역이 없다. 그런 말은 지시문에만 있다.
- 모델 출력은 어떤 정답 영역으로도 보정하지 않는다. 보류(`withheld`)는 유효한 결과다.

## 파일

| 파일 | 역할 |
|---|---|
| `config.yaml` | 지시문 3개, 조건 A, 이미지 레지스트리(해시, 역할), 생성 설정 |
| `prompts.yaml` | 프롬프트 본문(부모 v3에서 파생, 변경점은 파일 머리 주석에 기록) |
| `prompts_frozen.json` | 탐색 실행 전에 기록한 프롬프트 해시. 달라지면 실행을 거부한다 |
| `nla_probe.py` | 지시문 수준을 순회하는 실행기, 부모 스키마 + 필드 3개 |
| `nla_compare.py` | 수준별 비교표, 패널 이미지, "지시문이 줬다/모델이 덧붙였다" 정리 |
| `test_nla_offline.py` | GPU 없이 도는 검사 10건 |
| `data/images/` | 사진 3장 복사본 (부모와 해시 동일) |
| `results/<run_id>/` | 실행마다 새 폴더: `manifest.json`, `<이미지>/<수준>/{prompt.txt, raw_response.txt, structured.json, result.json, overlay_areas.png}`, `comparison/` |

## 실행

```bash
cd /home/binci/vlm_ws/semantic_task_navigation
PY=/home/binci/miniconda3/envs/stnav_gui/bin/python; export PYTHONDONTWRITEBYTECODE=1
$PY -m pytest -q experiments/natural_language_abstraction/test_nla_offline.py
$PY experiments/natural_language_abstraction/nla_probe.py prompts [--show]
$PY experiments/natural_language_abstraction/nla_probe.py run --split smoke          # D1: 출력 형식 확인 전용
$PY experiments/natural_language_abstraction/nla_probe.py freeze
$PY experiments/natural_language_abstraction/nla_probe.py run --split exploratory    # E1, E2 (동결 뒤에만)
$PY experiments/natural_language_abstraction/nla_compare.py experiments/natural_language_abstraction/results/<run_id>
```

## 실행 기록

| run | 용도 | 비고 |
|---|---|---|
| `20261118_185651_smoke_8a5b` | D1 × L1–L3, 출력 형식 확인 | L1이 스키마 `count ≥ 1` 때문에 무효 처리됨(모델이 `count: 0`을 적음). 검증 스키마만 `count ≥ 0`으로 고쳤고 프롬프트는 그대로 |
| `20261118_185930_exploratory_be08` | **E1, E2 × L1–L3 (6회)** | 프롬프트 `n0` 동결 후 1회씩 실행, 6건 모두 유효 |
