#!/usr/bin/env python3
"""Post-hoc (development) analysis: which deterministic Python gates separate the model's true pallet boxes from its
false ones? Usage: gate_analysis.py <run_dir> <probe>. Reads <run_dir>/evaluation.json and the photos; writes
<run_dir>/gate_analysis_<probe>.json. Gates were chosen AFTER looking at these 8 photos -> not an evaluation.
Features per detection (undistorted_px box): width, height, grey fraction (S<60, 110<V<235), red fraction
(H<10 or H>170, S>120, V>70).
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from evaluate import match, reference  # noqa: E402

PHOTO_DIR = Path("/home/binci/Pictures/Calibration")
PHOTOS = {1: "1calibration.png", 2: "2calibration.png", 3: "3calibration.png", 4: "4calibration.png",
          5: "5calibration.png", 6: "6calibration.png", 7: "7calibration.png",
          8: "undistorted_26007129092710.png"}
SIZE = (30, 75)          # px; reference pallets are 43-57 px
GREY_MIN = 0.35          # run 1 used 0.45 and rejected loaded pallets (the cube hides part of the plate): 0.38-0.47
CHANGE_LAB, CHANGE_MIN = 28, 0.30   # run 2: box must differ from the leave-one-out median background (other 7 photos)
RED_MIN_PX = 40          # red cube ~ 15-21 px side


def feats(hsv, b):
    x1, y1, x2, y2 = [int(round(v)) for v in b]
    c = hsv[max(y1, 0):y2, max(x1, 0):x2]
    if c.size == 0:
        return 0.0, 0
    h, s, v = c[..., 0], c[..., 1], c[..., 2]
    grey = ((s < 60) & (v > 110) & (v < 235)).mean()
    red = int((((h < 10) | (h > 170)) & (s > 120) & (v > 70)).sum())
    return float(grey), red


def change_frac(labs, p, b):
    x1, y1, x2, y2 = [int(round(v)) for v in b]
    bg = np.median(np.stack([labs[q][max(y1, 0):y2, max(x1, 0):x2] for q in labs if q != p]), axis=0)
    cur = labs[p][max(y1, 0):y2, max(x1, 0):x2]
    if cur.size == 0:
        return 0.0
    return float((np.linalg.norm(cur - bg, axis=2) > CHANGE_LAB).mean())


def main(run, probe):
    run = Path(run)
    ev = json.loads((run / "evaluation.json").read_text())
    ref = reference()
    hsvs = {p: cv2.cvtColor(cv2.imread(str(PHOTO_DIR / n)), cv2.COLOR_BGR2HSV) for p, n in PHOTOS.items()}
    labs = {p: cv2.cvtColor(cv2.imread(str(PHOTO_DIR / n)), cv2.COLOR_BGR2Lab).astype(np.float32)
            for p, n in PHOTOS.items()}
    rows = []
    for d in ev["details"]:
        if d["probe"] != probe:
            continue
        for det in d["dets"]:
            b = det["px"]
            lab = next((l for l in ref[d["photo"]] if match(b, l["box"])), None)
            g, r = feats(hsvs[d["photo"]], b)
            w, h = b[2] - b[0], b[3] - b[1]
            ch = change_frac(labs, d["photo"], b)
            keep = SIZE[0] <= w <= SIZE[1] and SIZE[0] <= h <= SIZE[1] and g >= GREY_MIN and ch >= CHANGE_MIN
            rows.append({"photo": d["photo"], "roi": d["roi"], "model_type": det["type"], "px": b,
                         "truth": lab["kind"] if lab else "FALSE", "w": round(w), "h": round(h),
                         "grey_frac": round(g, 2), "red_px": r, "change_frac": round(ch, 2), "gate_keep": keep,
                         "python_state": "loaded" if r >= RED_MIN_PX else "empty"})
    kept = [r for r in rows if r["gate_keep"]]
    # distinct reference objects recovered after the gates, with the state decided by the red-pixel count
    found = {}
    for r in kept:
        if r["truth"] != "FALSE":
            lab = next(l for l in ref[r["photo"]] if match(r["px"], l["box"]))
            found[(r["photo"], tuple(lab["box"]))] = found.get((r["photo"], tuple(lab["box"])), False) or \
                (r["python_state"] == r["truth"])
    total = sum(len(v) for v in ref.values())
    summary = {"probe": probe, "detections": len(rows), "false_before": sum(r["truth"] == "FALSE" for r in rows),
               "kept": len(kept), "false_after": sum(r["truth"] == "FALSE" for r in kept),
               "false_after_list": [(r["photo"], r["roi"], r["px"], r["w"], r["h"], r["grey_frac"], r["red_px"])
                                    for r in kept if r["truth"] == "FALSE"],
               "reference_objects": total, "reference_found_after": len(found),
               "reference_found_with_right_state": sum(found.values()),
               "true_rejected_by_gate": [(r["photo"], r["roi"], r["truth"], r["w"], r["h"], r["grey_frac"])
                                         for r in rows if r["truth"] != "FALSE" and not r["gate_keep"]],
               "gates": {"size_px": SIZE, "grey_min": GREY_MIN, "red_min_px": RED_MIN_PX, "change_lab": CHANGE_LAB, "change_min": CHANGE_MIN},
               "status": "POST_HOC_DEVELOPMENT (gates chosen after seeing these photos)"}
    (run / f"gate_analysis_{probe}.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=1))
    for r in rows:
        print(r["photo"], r["roi"], r["truth"], r["model_type"], r["w"], r["h"], r["grey_frac"], r["red_px"], r["change_frac"], "KEEP" if r["gate_keep"] else "")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
