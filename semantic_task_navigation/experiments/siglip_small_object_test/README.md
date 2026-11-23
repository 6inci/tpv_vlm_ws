# SigLIP2 작은 물체 식별 시험 (siglip_small_object_test)

**질문:** 이전과 같은 사진 8장, 같은 Grounding DINO 탐지 조건에서, Qwen 대신 SigLIP2가 작은 팔레트(회색 격자 받침)와 빨간 큐브를 맞게 식별하는가?

## 범위

| 항목 | 내용 |
|---|---|
| 탐지 | Grounding DINO (`grounding-dino-tiny`, 기존 v0.2 작업 프로세스, `qwen_vl` 환경) |
| 식별 | SigLIP2 (`siglip2-base-patch16-384`, 기존 `mm_worker`, `qwen_vl_q4` 환경, fp16). Qwen이 하던 일을 대신합니다. |
| Qwen 호출 | 없음 |
| 재사용 코드 | `multimodel_target_viewer`의 사진별 단계(`MMViewer.photo`)를 읽기 전용으로 import합니다.<br>DINO 문구, 임계값, SigLIP2 라벨, crop 방식은 그 설정 파일을 그대로 씁니다. |
| 환경 | 설치·다운로드 없음. 기존 Conda 환경과 캐시된 가중치를 그대로 씁니다. |
| 기존 파일 | 바꾸지 않습니다. 결과는 `results/<시각>_run/`에 새로 저장합니다. |

## 비교 기준 (AI 예비 판독, 사람이 확인한 정답 아님)

| 물체 | 개수 | 출처 |
|---|---|---|
| 빨간 큐브 (15~21 px) | 8 | `cube_candidate_funnel/config.yaml`의 `labels` |
| 큐브가 올라간 팔레트 (43~55 px) | 8 | STEP 4-A `survey.json`의 `cube.box_px` |
| 빈 팔레트 (42~51 px) | 7 | STEP 4-A `survey.json`의 `pallets_empty` (6번 사진은 없음) |

## 판정 규칙 (`config.yaml`, 실행 전에 선언)

| 판정 | 기준 |
|---|---|
| 탐지됨 (`detected`) | DINO 후보 중 하나가 기준 박스와 IoU 0.5 이상 (0.3 기준도 함께 기록) |
| 식별됨 (`IDENTIFIED`) | 탐지된 후보의 SigLIP2 1위 라벨이 맞는 묶음에 속함<br>• 팔레트: `a small pallet`, `a wooden pallet`<br>• 큐브: `a red cube` |
| 다른 이름 (`NAMED_OTHER`) | 탐지는 됐지만 1위 라벨이 다른 물체 |
| 미탐지 (`NOT_DETECTED`) | IoU 0.5 이상인 DINO 후보가 없음 |
| 기준 박스 식별 (`oracle`) | 기준 박스 자체를 SigLIP2에 넣은 결과. 탐지 실패와 이름 실패를 구분하려고 봅니다. |

- SigLIP2 점수는 설정의 라벨 24개 안에서의 상대 점수이며, 정답 확률이 아닙니다.
- SigLIP2가 팔레트나 큐브라고 부른 후보 중 기준 물체와 맞지 않는 것은 목록과 그림으로만 냅니다. 비교 기준에는 작은 팔레트와 큐브만 있으므로, 오탐률로 계산하지 않습니다. 적치 구역의 큰 팔레트는 실제 팔레트입니다.
- Qwen과의 직접 비교는 하지 않습니다. 저장된 Qwen 응답(`task_context_2d_navigation/results/20261119_174000`)에는 후보 박스 좌표가 없어서 같은 물체에 연결할 수 없습니다.

## 실행

```bash
cd /home/binci/vlm_ws/semantic_task_navigation
PY=/home/binci/miniconda3/envs/stnav_gui/bin/python
$PY -m pytest -q -p no:cacheprovider experiments/siglip_small_object_test        # 판정 로직 오프라인 검사
$PY experiments/siglip_small_object_test/run_test.py                              # 모델 실행 + 평가
$PY experiments/siglip_small_object_test/run_test.py --evaluate-only experiments/siglip_small_object_test/results/<run>   # 저장된 원자료로 평가만 다시
```

다른 모델 프로그램이 GPU를 쓰고 있을 때는 실행하지 마세요.

## 실행 기록 (2026-11-20, 모두 TESTING)

| 실행 | 내용 | 보고서 |
|---|---|---|
| `results/20261120_133027_run` | DINO + SigLIP2 base 이름 판정 (Qwen 대신) | `REPORT.md`, 수정 기준 재평가 `reeval_20261120_135602/` |
| `results/20261120_134533_followup_siglip_base_qwen` | 같은 crop으로 참조 이미지 비교, 라벨 문구 변경(진단), Qwen3-VL-4B 직접 비교 | `REPORT.md`, 수정 기준 재평가 `reeval_20261120_135604/` |
| `results/20261120_141616_followup_siglip_so400m` | 같은 crop으로 SigLIP2 so400m (이름 판정 + 참조 이미지 비교) | `REPORT.md` |

후속 실험 설정은 `config_followup.yaml`, 실행은 `run_followup.py --models siglip_base,qwen` 또는 `--models siglip_so400m`입니다.

- `--evaluate-only <실행 폴더>`는 기존 평가를 덮어쓰지 않고 `<실행 폴더>/reeval_<시각>/`에 저장합니다.
- 비교 기준 수정(사진 6의 빈 팔레트 추가)은 `config.yaml`의 `reference.additions`에 사유와 함께 기록했습니다.

## 결과 폴더

| 파일 | 내용 |
|---|---|
| `raw/<사진 ID>.json` | DINO 후보 전체, 후보별·기준 박스별 SigLIP2 순위 (평가는 이 원자료로만 다시 계산합니다) |
| `evaluation.json` | 물체별 판정과 요약 |
| `figures/reference_objects_gallery.png` | 기준 물체 23개: 기준 박스, 맞춘 DINO 박스, SigLIP2 상위 2개 |
| `figures/named_targets_photo_N.png` | SigLIP2가 팔레트·큐브라고 부른 DINO 후보 전체 |
| `session/` | `multimodel_target_viewer` 형식의 사진별 기록, SigLIP2에 넣은 crop, 모델 호출 기록, 작업 프로세스 로그 |
| `manifest.json`, `config_snapshot.yaml`, `reference_snapshot.json`, `run.log` | 실행 기록 |
