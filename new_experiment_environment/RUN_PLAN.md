# AUTO VLM Pipeline Run Plan

1. `EXP001_run01`의 Qwen 객체 탐지와 target-taxonomy region, `EXP002_semantic_region_eval`의 open semantic 결과를 원문 그대로 재사용한다.
2. 새 run에서 객체 bbox 기반 physical occupancy, 명시적 No-Go, image-plane floor/free-space를 분리 생성한다.
3. 기존 환경과 Hugging Face cache에서 Grounding DINO를 확인하고 최대 3회의 서로 다른 실행 시도만 한다. 실패하면 `BLOCKED`로 기록하고 Qwen 결과로 계속한다.
4. 가능한 detector 결과를 비교하고, 검증된 Qwen 객체·관계를 바탕으로 scene graph를 생성한다.
5. `IMAGE-PLANE / NON-METRIC` 2D map을 생성한다. 현재 입력과 일치하는 검증된 pixel↔world 자료가 없으면 metric map은 `BLOCKED`로 기록한다.
6. 한 번에 재실행 가능한 entry point, stage status, provenance, artifact manifest, `RUN_REPORT.md`를 생성하고 self-check로 검증한다.

제한: 기존 결과를 덮어쓰지 않으며 파일 삭제, package/system/CUDA/ROS2 변경, 실제 robot motion을 수행하지 않는다.
