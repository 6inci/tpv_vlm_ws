# Environment Inventory

Status: PASS

- Workspace: `/home/binci/vlm_ws/new_experiment_environment`
- Input: `/home/binci/vlm_ws/new_experiment_environment/approximate_environment.png` (1448×1086, SHA-256 `02a4e79589db3c15529b372768ef349b4ef2e4ebe2efb2454f605be43eb95a95`)
- Input audit: overlay가 없는 원본 작업환경 이미지로 확인됨.
- Reused Qwen objects/target regions: `/home/binci/vlm_ws/new_experiment_environment/results/EXP001_run01`
- Reused open semantic inference/evaluation: `/home/binci/vlm_ws/new_experiment_environment/results/EXP002_semantic_region_eval`
- Qwen model: `Qwen/Qwen3-VL-4B-Instruct`; preserved run device `cpu`
- Current Python/OpenCV/NumPy: `3.10.20` / `5.0.0` / `2.2.6`
- Current CUDA/GPU: `True` / `NVIDIA GeForce RTX 5070 Ti Laptop GPU`
- Grounding DINO package: 별도 package 없음; installed Transformers native implementation 사용 가능.
- Grounding DINO local cache before run: `False`
- Metric geometry: 이 입력 이미지와 일치하는 검증된 pixel↔world correspondence가 없음. 다른 카메라 장면의 homography는 재사용하지 않음.
- Safety: package/system/CUDA/ROS2 변경 및 robot motion 없음.
