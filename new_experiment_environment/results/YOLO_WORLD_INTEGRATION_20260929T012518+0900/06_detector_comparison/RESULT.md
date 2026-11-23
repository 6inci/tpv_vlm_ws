# Qwen / Grounding DINO / YOLO-World Comparison

Status: PARTIAL

Ground truth가 없어 accuracy를 계산하지 않았다. Qwen 31개, Grounding DINO 58개, YOLO-World 39개다. YOLO-World의 Qwen-compatible match는 19개, Grounding-DINO-compatible match는 27개다.

- Conveyor detected: `False`
- Manipulator detected: `False`
- Left manipulator count (x-center < 650 px): `0`
- Central manipulator count (650 <= x-center < 1000 px): `0`
- Duplicate criterion: same canonical label and bbox IoU >= 0.50
