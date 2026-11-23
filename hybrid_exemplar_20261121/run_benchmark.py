#!/usr/bin/env python3
"""Same 16 conditions as the 11-20 benchmark (8 photos x I1, I2), hybrid pipeline, scored by the UNCHANGED 11-20 scorer.

  /home/binci/miniconda3/envs/stnav_gui/bin/python -B run_benchmark.py [--photos all] [--instructions I1,I2]
Writes only to session_<stamp>/ in this folder. The pipeline never reads ground_truth.yaml; only evaluate_tasks does, after the runs.
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
BENCH = Path("/home/binci/vlm_ws/model_performance_20261120")       # ground_truth.yaml + evaluate_tasks.py (read only)
sys.path.insert(0, str(HERE))
import hybrid_pipeline as HP  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photos", default="all")
    ap.add_argument("--instructions", default="I1,I2")
    a = ap.parse_args()
    gt = yaml.safe_load((BENCH / "ground_truth.yaml").read_text(encoding="utf-8"))
    session = HERE / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    v = HP.HybridViewer(session_dir=session)
    photos = {p["photo_id"]: p for p in v.list_photos()}
    wanted = list(gt["photos"]) if a.photos == "all" else a.photos.split(",")
    assert not [p for p in wanted if p not in photos], "photo not found by content hash"
    plan = {"started": datetime.now().isoformat(timespec="seconds"), "backends": [HP.CFG["vlm_backend"]], "photos": wanted,
            "instructions": {k: gt["instructions"][k]["text"] for k in a.instructions.split(",")}, "rules": HP.CFG["rules"],
            "pipeline": str(HERE / "hybrid_pipeline.py"), "config": str(HERE / "config.yaml"), "session": str(session)}
    (session / "benchmark_plan.json").write_text(json.dumps(plan, indent=1, ensure_ascii=False), encoding="utf-8")
    index = []
    try:
        for pid in wanted:
            for ik, text in plan["instructions"].items():
                t0 = time.time()
                rec = v.run(photos[pid]["path"], text)
                row = {"backend": HP.CFG["vlm_backend"], "photo_id": pid, "instruction_key": ik, "run_id": rec["run_id"], "wall_s": rec["wall_s"],
                       "first_failure": rec["first_failure"], "A_status": rec["source"].get("status"), "B_status": rec["destination"].get("status"),
                       "timing_s": rec["timing_s"]}
                index.append(row)
                (session / "benchmark_index.json").write_text(json.dumps(index, indent=1, ensure_ascii=False), encoding="utf-8")
                print(json.dumps(row, ensure_ascii=False), f"({time.time() - t0:.0f} s)", flush=True)
    finally:
        v.close()
    plan["finished"] = datetime.now().isoformat(timespec="seconds")
    (session / "benchmark_plan.json").write_text(json.dumps(plan, indent=1, ensure_ascii=False), encoding="utf-8")
    sys.path.insert(0, str(BENCH))
    import evaluate_tasks as E   # writes task_evaluation.json / .md into the session only
    E.main(session)
    print("session:", session)


if __name__ == "__main__":
    sys.exit(main())
