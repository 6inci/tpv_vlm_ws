#!/usr/bin/env python3
"""Task-level benchmark of the CURRENT model pipeline (multimodel_target_viewer: Grounding DINO -> SigLIP2 -> Qwen3-VL -> Python).

The pipeline code is imported read-only (MMViewer); nothing in semantic_task_navigation is modified. All outputs go to
/home/binci/vlm_ws/model_performance_20261120/session_<stamp>/ (passed as session_dir). Run with python -B so no .pyc is
written next to the existing code.

Usage: python -B run_benchmark.py [--backends qwen3vl_8b_nf4,qwen3vl_4b_bf16] [--photos all] [--instructions I1,I2]
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
MMV = Path("/home/binci/vlm_ws/semantic_task_navigation/experiments/multimodel_target_viewer")
sys.path.insert(0, str(MMV))
import mm_pipeline as M  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", default="qwen3vl_8b_nf4,qwen3vl_4b_bf16")
    ap.add_argument("--photos", default="all")
    ap.add_argument("--instructions", default="I1,I2")
    ap.add_argument("--rules", default="R1")
    a = ap.parse_args()
    gt = yaml.safe_load((HERE / "ground_truth.yaml").read_text(encoding="utf-8"))
    session = HERE / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    v = M.MMViewer(session_dir=session)
    photos = {p["photo_id"]: p for p in v.list_photos()}
    wanted = list(gt["photos"]) if a.photos == "all" else a.photos.split(",")
    missing = [p for p in wanted if p not in photos]
    assert not missing, f"photos not found by content hash: {missing}"
    plan = {"started": datetime.now().isoformat(timespec="seconds"), "backends": a.backends.split(","), "photos": wanted,
            "instructions": {k: gt["instructions"][k]["text"] for k in a.instructions.split(",")}, "rules": a.rules,
            "pipeline": str(MMV), "pipeline_config": str(MMV / "config.yaml"), "session": str(session)}
    (session / "benchmark_plan.json").write_text(json.dumps(plan, indent=1, ensure_ascii=False), encoding="utf-8")
    index = []
    try:
        for backend in plan["backends"]:
            for pid in wanted:
                for ik, text in plan["instructions"].items():
                    t0 = time.time()
                    rec = v.run(photos[pid]["path"], text, a.rules, backend=backend)
                    row = {"backend": backend, "photo_id": pid, "instruction_key": ik, "run_id": rec["run_id"],
                           "wall_s": rec["wall_s"], "first_failure": rec["first_failure"],
                           "A_status": rec["source"].get("status"), "B_status": rec["destination"].get("status")}
                    index.append(row)
                    (session / "benchmark_index.json").write_text(json.dumps(index, indent=1, ensure_ascii=False), encoding="utf-8")
                    print(json.dumps(row, ensure_ascii=False), f"({time.time() - t0:.0f} s)", flush=True)
    finally:
        v.close()
    plan["finished"] = datetime.now().isoformat(timespec="seconds")
    (session / "benchmark_plan.json").write_text(json.dumps(plan, indent=1, ensure_ascii=False), encoding="utf-8")
    print("session:", session)


if __name__ == "__main__":
    sys.exit(main())
