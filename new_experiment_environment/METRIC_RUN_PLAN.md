# Synthetic Metric Pipeline Run Plan

1. 새 1920×1080 입력과 기존 Qwen/Grounding DINO 코드·cache를 사용해 fresh detector evidence를 만든다.
2. 8개 calibration/4개 holdout point로 synthetic mat homography와 residual을 계산한다.
3. Grounding DINO bbox metric map을 만든 뒤, installed Transformers의 SAM2로 mask occupancy를 생성·비교한다.
4. YOLO-World는 기존 설치/cache만 사용하며, 승인 없는 package 설치가 필요하면 `BLOCKED`로 기록한다.
5. semantic/detection/segmentation/geometry layer를 metric scene graph와 1 cm/px feasibility map으로 통합한다.
6. stdlib read-only MCP prototype을 생성·self-check하되 외부 client 검증 전에는 `PARTIAL`로 표시한다.
7. 기존 run을 건드리지 않고 새 run의 `RUN_REPORT.md`와 manifest까지 검증한다.
