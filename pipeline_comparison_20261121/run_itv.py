#!/usr/bin/env python3
"""instruction_target_viewer (11-19: task_context pipeline = Qwen3-VL-4B + Grounding DINO + Qwen role judgement + Python policy)
on the same 8 photos x 2 instructions. Imported read-only; nothing in semantic_task_navigation is modified. Run with python -B.

Two in-memory adaptations (no file changes), both needed only because the photos were renamed on 2026-11-19
(multimodel_target_viewer README, "사진 이름 변경"); they reproduce what multimodel_target_viewer does by content hash:
  1. photo ids by sha256 (VP.photo_id is replaced by the hash -> id map of cube_candidate_funnel/config.yaml)
  2. the E1 floor-transform image path is set to the photo whose sha256 equals the recorded rcfg['image_sha256']
"""
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
EXP = Path("/home/binci/vlm_ws/semantic_task_navigation/experiments")
sys.path.insert(0, str(EXP / "instruction_target_viewer"))
import viewer_pipeline as VP  # noqa: E402

PHOTO_DIR = Path("/home/binci/Pictures/Calibration")
SHA2ID = {i["sha256"]: i["id"] for i in yaml.safe_load((EXP / "cube_candidate_funnel/config.yaml").read_text(encoding="utf-8"))["images"]}
_cache = {}


def sha(p):
    p = str(p)
    if p not in _cache:
        _cache[p] = hashlib.sha256(Path(p).read_bytes()).hexdigest()
    return _cache[p]


VP.photo_id = lambda path: SHA2ID.get(sha(path), Path(path).stem)   # adaptation 1


def main():
    gt = yaml.safe_load((HERE / "ground_truth_COPY.yaml").read_text(encoding="utf-8"))
    session = HERE / f"itv_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    v = VP.Viewer(session_dir=session)
    e1 = next(p for p in PHOTO_DIR.glob("*.png") if sha(p) == v.rcfg["image_sha256"])
    v.rcfg["image"] = str(e1)                                          # adaptation 2
    paths = {SHA2ID.get(sha(p)): str(p) for p in PHOTO_DIR.glob("*.png")}
    index = []
    try:
        for pid in gt["photos"]:
            for ik in ("I1", "I2"):
                t0 = time.time()
                rec = v.run(paths[pid], gt["instructions"][ik]["text"], "R1")
                row = {"backend": "itv_qwen3vl_4b", "photo_id": pid, "instruction_key": ik, "run_id": rec["run_id"], "wall_s": rec.get("wall_s"),
                       "first_failure": rec.get("first_failure"), "A_status": rec["source"].get("status"), "B_status": rec["destination"].get("status")}
                index.append(row)
                (session / "benchmark_index.json").write_text(json.dumps(index, indent=1, ensure_ascii=False), encoding="utf-8")
                print(json.dumps(row, ensure_ascii=False), f"({time.time() - t0:.0f} s)", flush=True)
    finally:
        v.close()
    print("session:", session, flush=True)


if __name__ == "__main__":
    sys.exit(main())
