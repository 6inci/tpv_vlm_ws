#!/usr/bin/env python3
"""Score saved runs against a human ground-truth file (section 14: each item scored independently).

  python scripts/evaluate_run.py --gt <ground_truth.json> runs/<run_id> [runs/<run_id> ...] [--iou 0.5]

Ground truth (written by a person, never by the system):
{
  "image_sha256": "<sha256 of the source image>",  "annotator": "...",  "split": "dev" | "test",
  "task": {"task_type": "...", "target": {"class": "cube", "color": "red", "source_reference": "conveyor"},
           "destination": {"region_type": "loading_area", "reference_object": "truck"}},
  "objects": [{"class_name": "conveyor", "bbox_xyxy": [x1, y1, x2, y2]}, ...],      # raw_image_px
  "target_bbox_xyxy": [x1, y1, x2, y2],
  "loading_region_polygon": [[x, y], ...],
  "placement_region_polygons": [[[x, y], ...], ...]                                 # v0.2 (GUI '정답: 배치영역')
}
Reference files saved by the GUI (data/reference/*.json) use the same keys; missing keys are NO_GT (pending).
Writes runs/<run_id>/evaluation/eval_<timestamp>.json (new file; the run itself is not modified).
Items computed from manual_test_data (IMAGE_DEMO) are reported NOT_APPLICABLE, never as successes.
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from semantic_task_nav.artifacts import dumps  # noqa: E402
from semantic_task_nav.geometry.coordinate_frames import iou  # noqa: E402
from semantic_task_nav.semantics.region_grounding import polygon_iou  # noqa: E402


def union_mask(polys, size):
    w, h = size
    m = np.zeros((h, w), np.uint8)
    for p in polys:
        cv2.fillPoly(m, [np.round(np.array(p)).astype(np.int32)], 1)
    return m.astype(bool)


def score(result, gt, thr):
    out = {"run_id": result["run_id"], "analysis_mode": result["analysis_mode"], "iou_threshold": thr}
    if gt.get("image_sha256") and gt["image_sha256"] != result["frame"].get("source_sha256"):
        out["error"] = "ground truth belongs to a different image"
        return out
    manual = result["analysis_mode"] == "IMAGE_DEMO"
    # 14.1 task parsing
    task, gtt = result.get("task"), gt.get("task")
    if gtt is None:
        out["task_parsing"] = "NO_GT"
    else:
        ok = task is not None and task["task_type"] == gtt["task_type"] and all(
            (task.get(k) or {}).get(f) == v for k in ("target", "destination") for f, v in (gtt.get(k) or {}).items())
        out["task_parsing"] = {"correct": ok, "backend": (task or {}).get("parser", {}).get("backend")}
    # 14.2 object grounding
    objs = [o for o in (result.get("detections") or {}).get("objects", []) if o.get("accepted")]
    tgt = (result.get("target_inspection") or {}).get("target")
    if manual:
        out["object_grounding"] = "NOT_APPLICABLE (manual_test_data)"
    else:
        per = {}
        for g in gt.get("objects", []):
            best = max([iou(o["bbox_xyxy"], g["bbox_xyxy"]) for o in objs if o["class_name"] == g["class_name"]] or [0.0])
            per[g["class_name"]] = {"best_iou": round(best, 3), "hit": best >= thr}
        g = gt.get("target_bbox_xyxy")
        t_iou = iou(tgt["bbox_xyxy"], g) if tgt and g else 0.0
        # small objects: IoU is harsh for a few-pixel offset, so the centre-in-box criterion is reported too
        centre_in = bool(tgt and g and g[0] <= (tgt["bbox_xyxy"][0] + tgt["bbox_xyxy"][2]) / 2 <= g[2]
                         and g[1] <= (tgt["bbox_xyxy"][1] + tgt["bbox_xyxy"][3]) / 2 <= g[3])
        out["object_grounding"] = {"per_class": per, "target_iou": round(t_iou, 3), "target_hit": t_iou >= thr,
                                   "target_centre_in_gt_box": centre_in,
                                   "target_status": (result.get("target_inspection") or {}).get("status")}
    # 14.3 semantic grounding (v0.2: destination.semantic_loading_region; v0.1: region_grounding)
    rg = result.get("region_grounding") or {}
    dest = result.get("destination") or {}
    reg = dest.get("semantic_loading_region") if dest else rg.get("selected_loading_region")
    w, h = result["frame"]["width"], result["frame"]["height"]
    if manual:
        out["semantic_grounding"] = "NOT_APPLICABLE (manual_test_data)"
    elif gt.get("loading_region_polygon"):
        r_iou = polygon_iou(reg["polygon_xy"], gt["loading_region_polygon"], (w, h)) if reg else 0.0
        out["semantic_grounding"] = {"status": rg.get("status"), "iou": round(r_iou, 3), "hit": r_iou >= 0.3,
                                     "note": "IoU >= 0.3 with the person-drawn loading region"}
    else:
        out["semantic_grounding"] = "NO_GT (pending human annotation)"
    # v0.2 destination chain: placement region and robot goal against the person-drawn placement region
    place = dest.get("object_placement_region") if isinstance(dest.get("object_placement_region"), dict) else None
    goal = dest.get("selected_robot_goal")
    if not gt.get("placement_region_polygons"):
        out["placement_and_goal"] = "NO_GT (pending human annotation)"
    else:
        gm = union_mask(gt["placement_region_polygons"], (w, h))
        pm = union_mask(place["polygons_xy"], (w, h)) if place and place.get("polygons_xy") else np.zeros_like(gm)
        union = np.count_nonzero(gm | pm)
        dist = None
        if goal:
            d = cv2.distanceTransform((~gm).astype(np.uint8), cv2.DIST_L2, 5)
            dist = round(float(d[int(goal["xy"][1]), int(goal["xy"][0])]), 1)
        reach = ((dest.get("robot_approach_region") or {}).get("reach") or {})
        out["placement_and_goal"] = {
            "final_stage_status": dest.get("final_stage_status"), "failed_stage": dest.get("failed_stage"),
            "placement_iou": round(np.count_nonzero(gm & pm) / union, 3) if union else 0.0,
            "goal_to_gt_placement_px": dist, "reach_px": reach.get("reach_px"), "reach_status": reach.get("status"),
            "note": "distance is image px; with PROVISIONAL reach this is not a physical feasibility result"}
    # 14.4 goal selection / 14.5 path planning (image space)
    goals, val = result.get("goals") or {}, result.get("validation") or {}
    out["goal_selection"] = {k: (goals.get(k) or {}).get("status") for k in ("pickup", "dropoff")}
    out["path_planning"] = {k: {"status": p["status"], "valid": (val.get(k) or {}).get("valid"),
                                "min_clearance_px": (val.get(k) or {}).get("min_clearance_px")}
                            for k, p in (result.get("paths") or {}).items()}
    # 14.6 end to end
    e2e_ok = (not manual and result["overall_status"] == "PLAN_COMPLETE"
              and isinstance(out.get("task_parsing"), dict) and out["task_parsing"]["correct"]
              and isinstance(out.get("object_grounding"), dict)
              and (out["object_grounding"]["target_hit"] or out["object_grounding"]["target_centre_in_gt_box"])
              and isinstance(out.get("semantic_grounding"), dict) and out["semantic_grounding"]["hit"])
    out["end_to_end_planning"] = {"overall_status": result["overall_status"], "automatic_and_correct": e2e_ok,
                                  "uses_manual_inputs": result["evaluation_record"]["end_to_end_planning"]["uses_manual_inputs"],
                                  "note": "image-space planning only; not robot execution"}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    gt = json.loads(Path(a.gt).read_text(encoding="utf-8"))
    for run in a.runs:
        result = json.loads((Path(run) / "result.json").read_text(encoding="utf-8"))
        s = score(result, gt, a.iou)
        s["ground_truth_file"] = str(Path(a.gt).resolve())
        out = Path(run) / "evaluation" / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(dumps(s), encoding="utf-8")
        print(dumps(s))
        print(f"-> {out}")


if __name__ == "__main__":
    main()
