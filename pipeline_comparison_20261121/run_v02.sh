#!/usr/bin/env bash
# v0.2 (semantic_task_navigation run.py, unchanged) on the same 8 photos x 2 instructions.
# Outputs are redirected into this folder with --set app.runs_dir / app.worker_log_dir (recorded in each run manifest).
# -B: no .pyc written next to the existing code.
set -u
OUT=/home/binci/vlm_ws/pipeline_comparison_20261121/v02_runs
mkdir -p "$OUT"
PY=/home/binci/miniconda3/envs/stnav_gui/bin/python
cd /home/binci/vlm_ws/semantic_task_navigation
I1="빨간 큐브를 비어 있는 팔레트로 옮겨줘."
I2="하역장 옆에 있는 물체를 컨베이어 벨트 옆에 놓아줘."
for f in 1calibration.png 2calibration.png 3calibration.png 4calibration.png 5calibration.png 6calibration.png 7calibration.png undistorted_26007129092710.png; do
  for k in I1 I2; do
    cmd=$([ "$k" = I1 ] && echo "$I1" || echo "$I2")
    echo "=== $f $k"
    $PY -B run.py --mode image --headless --analysis-mode LOCAL_ONLY --image "/home/binci/Pictures/Calibration/$f" --command "$cmd" \
      --set "app.runs_dir=$OUT" --set "app.worker_log_dir=$OUT/_worker_logs" 2>&1 | tail -n 25
  done
done
