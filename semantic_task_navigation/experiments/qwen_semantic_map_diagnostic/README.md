# Qwen3-VL 의미 지도 진단 실험 (2026-11-18)

외부 고정 카메라 사진 **1장**에서 Qwen3-VL-4B가 객체, 이동로봇/적치물 구분, 공간 관계, 기능 영역, 자연어 목적지를 한 번의 응답으로 얼마나 일관된 의미 지도로 표현하는지 본 **탐색적 진단**. 결과: [REPORT_2026-11-18.md](REPORT_2026-11-18.md)

- 정확도, IoU, 성공률은 계산하지 않는다(정답 데이터 없음). "AI 예비 검토"는 AI가 이미지를 보고 한 판단이다.
- Qwen3-VL-4B-Instruct만 썼다(기존 로컬 모델·worker 그대로). 쓰지 않은 것: Grounding DINO, SAM2, YOLO-World, V1/V2 검증기, A\*, ROS2, 하드웨어. 추론은 1회.
- 기존 v0.2 코드, 이전 실험의 코드·결과·동결 프롬프트는 수정하지 않았다.

## 파일

| 파일 | 내용 |
|---|---|
| `image_selection.json` | 이미지 선정 기록 (**추론 전에 기록**, 8장 후보 해시·해상도·규칙) |
| `reviewer_inventory_pre_inference.json` | 추론 전에 AI가 작성한 객체 목록(예비 기준). 모델에 전달하지 않음 |
| `prompts.yaml`, `prompt_frozen.json` | 새 프롬프트 `sm0`와 실행 전 동결 해시 (`superseded/`: 모델이 보기 전에 폐기한 첫 동결 파일) |
| `config.yaml` | 지시문, 이미지, 생성 설정 |
| `semantic_map_probe.py` | 실행기 (Qwen 호출 1회, 스키마 검증, 불완전 응답 복구, 매니페스트) |
| `visualize.py` | 객체 지도 / 의미 영역 지도 / 통합 비교 / 검토용 크롭 |
| `test_probe_offline.py` | GPU 없이 도는 검사 12건 |
| `data/images/` | 선정한 사진 사본 (원본 수정 없음) |
| `results/20261118_223555_3476/` | 실행 결과: `prompt.txt`, `raw_response.txt`, `structured.json`, `result.json`, `manifest.json`, `01_object_map.png`, `02_semantic_area_map.png`, `03_comparison.png`, `review/` |

## 실행

```bash
cd /home/binci/vlm_ws/semantic_task_navigation
PY=/home/binci/miniconda3/envs/stnav_gui/bin/python; export PYTHONDONTWRITEBYTECODE=1
$PY -m pytest -q experiments/qwen_semantic_map_diagnostic/test_probe_offline.py
$PY experiments/qwen_semantic_map_diagnostic/semantic_map_probe.py prompts [--show]
$PY experiments/qwen_semantic_map_diagnostic/semantic_map_probe.py freeze     # 이미 동결됨
$PY experiments/qwen_semantic_map_diagnostic/semantic_map_probe.py run        # 같은 이미지+프롬프트의 재실행은 --allow-repeat 필요
$PY experiments/qwen_semantic_map_diagnostic/semantic_map_probe.py render results/<run>   # 모델 호출 없이 그림만 다시 그림
```
