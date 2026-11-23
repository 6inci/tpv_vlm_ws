"""Robot start from a DETECTION, not from a typed coordinate.

1. Stored Grounding DINO "mobile robot." candidates (v0.2 run, nothing re-detected) whose centre lies in the coarse window given in the task text.
2. v0.2 Qwen crop check (grounding.verify_box, unchanged) in descending detector score; the first "yes" is the robot. v0.2 had skipped
   these candidates only because of verify_top_k=3 (higher-scoring boxes were robot arms); this is the one new model call of this step.
3. The detector box is an image OUTLINE, not a floor footprint. The footprint and the start point are therefore recorded as estimates
   with their method and uncertainty (see start_estimates)."""
import math
from pathlib import Path

import numpy as np


def stored_candidates(detections, window, cls="mobile_robot"):
    x1, y1, x2, y2 = window
    out = []
    for o in detections["objects"]:
        if o["class_name"] != cls:
            continue
        b = o["bbox_xyxy"]
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            out.append(o)
    return sorted(out, key=lambda o: -(o.get("score") or 0))


def verify_in_order(client, mcfg, image, cands, crop_dir, verify_box):
    """Returns (accepted candidate | None, [records]). Stops at the first 'yes'."""
    desc = mcfg["verification"]["descriptions"]["mobile_robot"]
    recs = []
    for i, c in enumerate(cands, 1):
        v = verify_box(client, mcfg, image, c["bbox_xyxy"], desc, Path(crop_dir) / f"verify_robot_{c['object_id']}.png")
        recs.append({"candidate_id": c["object_id"], "bbox_xyxy": c["bbox_xyxy"], "detector_score": c.get("score"),
                     "detector_prompt": c.get("prompt"), "status": v["status"], "answer": v["answer"], "crop": v["crop"],
                     "call": v["call"], "stored_verification_in_v02_run": c.get("verification", {})})
        if v["status"] == "yes":
            return c, recs
    return None, recs


def start_estimates(box, principal_point, shifts_px):
    """The outline centre is displaced away from the principal point for a tall object (parallax), so the floor point below the robot
    lies on the line outline-centre -> principal point. Each shift gives one candidate start; shift 0 is the plain outline centre.
    Neither is a measured floor position (no validated ground homography exists)."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    d = np.array(principal_point, float) - np.array([cx, cy])
    u = d / np.linalg.norm(d)
    out = []
    for s in shifts_px:
        out.append({"shift_px": s, "xy": [round(float(cx + u[0] * s), 1), round(float(cy + u[1] * s), 1)],
                    "method": "outline centre" if s == 0 else f"outline centre moved {s} px towards the principal point"})
    return out, [round(float(u[0]), 3), round(float(u[1]), 3)]


def footprint_record(box, v02_radius_px):
    w, h = box[2] - box[0], box[3] - box[1]
    return {"observed_outline_box_xyxy": [round(v, 1) for v in box], "outline_size_wh_px": [round(w, 1), round(h, 1)],
            "occupancy_upper_bound": "the outline box (the floor footprint lies inside the visible outline up to parallax)",
            "planning_discs": {
                "assumed_v02_radius_px": {"radius_px": v02_radius_px, "status": "ASSUMED",
                                          "note": "planning.yaml default; not measured; smaller than the robot's visible outline"},
                "image_inscribed_radius_px": {"radius_px": int(math.floor(min(w, h) / 2)), "status": "ESTIMATED_FROM_IMAGE_OUTLINE",
                                              "note": "circle inscribed in the verified outline box: stress test only, still not a rectangle-aware footprint"}},
            "floor_position": "NOT CONFIRMED: no validated ground-plane transform; start is an estimate with the listed uncertainty"}
