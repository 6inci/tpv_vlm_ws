"""Semantic destination: truck object -> loading-area candidates -> VLM selection -> region.

Kept separate on purpose:
  truck_object             detected + verified truck box
  loading_area_candidates  deterministic geometric bands around the truck (algorithm output)
  selected_loading_region  candidate chosen by the VLM among numbered candidates (unverified inference)
  validated_loading_region only when it matches a CONFIRMED prior zone; otherwise null
  robot_accessible_region / selected_dropoff_pose  -> navigation.goal_selector
The VLM chooses an ID; it never supplies coordinates.
"""
import cv2
import numpy as np

from ..contracts import RAW_PX, REGION_SCHEMA, validate
from ..perception.grounding import qwen_json

SELECT_SCHEMA = {
    "type": "object", "required": ["candidates", "best", "evidence", "uncertain"],
    "properties": {"candidates": {"type": "array", "items": {
                       "type": "object", "required": ["id", "plausible"],
                       "properties": {"id": {"type": "integer"},
                                      "plausible": {"type": "string", "enum": ["yes", "no", "uncertain"]},
                                      "reason": {"type": ["string", "null"]}}}},
                   "best": {"type": ["integer", "null"]},
                   "evidence": {"type": "array", "items": {"type": "string"}},
                   "uncertain": {"type": "array", "items": {"type": "string"}}},
}
SELECTION_RULE = ("1) the VLM's 'best' if it is judged plausible (yes) and robot-accessible; 2) otherwise the single "
                  "other candidate that is plausible and accessible; 3) otherwise no region (uncertain/inaccessible). "
                  "The VLM never supplies coordinates; accessibility is computed on the navigation map.")


def generate_candidates(truck_box, image_size, rcfg):
    w, h = image_size
    x1, y1, x2, y2 = truck_box
    d = max(rcfg["min_band_px"], rcfg["band_ratio"] * max(x2 - x1, y2 - y1))
    bands = {"below": [x1, y2, x2, y2 + d], "above": [x1, y1 - d, x2, y1],
             "left": [x1 - d, y1, x1, y2], "right": [x2, y1, x2 + d, y2]}
    out = []
    for side, (a, b, c, e) in bands.items():
        a, b, c, e = max(0, a), max(0, b), min(w, c), min(h, e)
        if (c - a) * (e - b) < rcfg["min_candidate_area_px"]:
            continue
        out.append({"candidate_id": len(out) + 1, "side_of_truck_box": side,
                    "polygon_xy": [[round(a, 1), round(b, 1)], [round(c, 1), round(b, 1)],
                                   [round(c, 1), round(e, 1)], [round(a, 1), round(e, 1)]],
                    "coordinate_frame": RAW_PX,
                    "generation": f"axis-aligned band of depth {round(d, 1)} px adjacent to the truck box"})
    return out


def render_selection_input(image, truck_box, candidates, margin):
    h, w = image.shape[:2]
    pts = np.array([p for c in candidates for p in c["polygon_xy"]] + [[truck_box[0], truck_box[1]],
                                                                      [truck_box[2], truck_box[3]]])
    x0, y0 = int(max(0, pts[:, 0].min() - margin)), int(max(0, pts[:, 1].min() - margin))
    x1, y1 = int(min(w, pts[:, 0].max() + margin)), int(min(h, pts[:, 1].max() + margin))
    crop = image[y0:y1, x0:x1].copy()
    pal = [(0, 200, 255), (255, 120, 255), (0, 255, 120), (255, 200, 0)]
    for c in candidates:
        poly = (np.array(c["polygon_xy"]) - [x0, y0]).astype(np.int32)
        col = pal[(c["candidate_id"] - 1) % len(pal)]
        cv2.polylines(crop, [poly], True, col, 2)
        cx, cy = poly.mean(axis=0).astype(int)
        cv2.putText(crop, str(c["candidate_id"]), (cx - 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5)
        cv2.putText(crop, str(c["candidate_id"]), (cx - 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2)
    tb = [int(truck_box[0] - x0), int(truck_box[1] - y0), int(truck_box[2] - x0), int(truck_box[3] - y0)]
    cv2.rectangle(crop, tb[:2], tb[2:], (255, 255, 255), 2)
    cv2.putText(crop, "T", (tb[0] + 4, tb[1] + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    scale = max(1.0, 896 / max(crop.shape[:2]))
    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return crop, {"crop_xyxy_raw": [x0, y0, x1, y1], "scale": round(scale, 4)}


def _region(cand, truck_id, evidence, status, source):
    return validate({"region_id": f"loading_area_candidate_{cand['candidate_id']}", "region_type": "loading_area",
                     "polygon_xy": cand["polygon_xy"], "coordinate_frame": RAW_PX, "reference_objects": [truck_id],
                     "evidence": evidence, "verification_status": status, "source": source}, REGION_SCHEMA)


def polygon_iou(poly_a, poly_b, size):
    w, h = size
    a, b = np.zeros((h, w), np.uint8), np.zeros((h, w), np.uint8)
    cv2.fillPoly(a, [np.round(np.array(poly_a)).astype(np.int32)], 1)
    cv2.fillPoly(b, [np.round(np.array(poly_b)).astype(np.int32)], 1)
    union = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b) / union) if union else 0.0


def ground_loading_region(client, mcfg, pcfg, image, truck, out_dir, prior_zones=(), accessible_cells=None):
    """accessible_cells(polygon) -> number of collision-free robot-centre cells within the drop-off reach
    (None when no navigation map is available)."""
    rcfg = pcfg["region_grounding"]
    h, w = image.shape[:2]
    cands = generate_candidates(truck["bbox_xyxy"], (w, h), rcfg)
    res = {"truck_object": truck["object_id"], "loading_area_candidates": cands, "selection": None,
           "selected_loading_region": None, "validated_loading_region": None, "prior_zone_consistency": []}
    if not cands:
        res["status"] = "SEMANTIC_REGION_UNCERTAIN"
        res["reason"] = "no candidate band inside the image"
        return res
    crop, crop_info = render_selection_input(image, truck["bbox_xyxy"], cands, rcfg["context_margin_px"])
    path = out_dir / "region_select_input.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), crop)
    ids = [c["candidate_id"] for c in cands]
    prompt = mcfg["prompts"]["region_select"]["text"].replace("{candidate_ids}", ", ".join(map(str, ids)))
    obj, rec = qwen_json(client, mcfg, "region_select", prompt, path, mcfg["qwen"]["max_new_tokens"]["region_select"],
                         SELECT_SCHEMA)
    res["selection"] = {"method": "vlm_plausibility_per_numbered_geometric_candidate + robot_accessibility",
                        "rule": SELECTION_RULE, "input_image": str(path), "input_crop": crop_info, "call": rec,
                        "output": obj}
    if obj is None:
        res["status"] = "SEMANTIC_REGION_UNCERTAIN"
        res["reason"] = rec["error"]
        return res
    judged = {c["id"]: c for c in obj["candidates"] if c["id"] in ids}
    for c in cands:
        j = judged.get(c["candidate_id"])
        c["vlm_plausible"] = j["plausible"] if j else "not_judged"
        c["vlm_reason"] = j.get("reason") if j else None
        c["accessible_cells"] = None if accessible_cells is None else int(accessible_cells(c["polygon_xy"]))
    res["model_best"] = obj["best"]
    res["model_uncertainty"] = obj["uncertain"]
    ok = [c for c in cands if c["vlm_plausible"] == "yes" and (c["accessible_cells"] is None or c["accessible_cells"] > 0)]
    best = next((c for c in ok if c["candidate_id"] == obj["best"]), None)
    if best is None and len(ok) == 1:
        best = ok[0]
    if best is None:
        plausible = [c["candidate_id"] for c in cands if c["vlm_plausible"] == "yes"]
        if plausible and not ok:
            res["status"] = "SEMANTIC_REGION_INACCESSIBLE"
            res["reason"] = (f"candidates judged plausible {plausible} have no collision-free robot position within "
                             f"reach; VLM best={obj['best']}")
        else:
            res["status"] = "SEMANTIC_REGION_UNCERTAIN"
            res["reason"] = (f"plausible+accessible candidates: {[c['candidate_id'] for c in ok]}; VLM best="
                             f"{obj['best']} (no unique choice)")
        return res
    evidence = list(obj["evidence"]) + ([f"candidate {best['candidate_id']}: {best['vlm_reason']}"]
                                        if best.get("vlm_reason") else [])
    region = _region(best, truck["object_id"], evidence, "unverified", "geometric_candidate+vlm_plausibility+accessibility")
    res["selected_loading_region"] = region
    res["status"] = "REGION_SELECTED_UNVERIFIED"
    res["selected_via"] = "vlm_best" if best["candidate_id"] == obj["best"] else "only_plausible_accessible_candidate"
    for z in prior_zones:  # evaluation/consistency only; never feeds planning
        res["prior_zone_consistency"].append({"zone_id": z["id"], "role": z["role"], "confirmed": z["confirmed"],
                                              "iou": round(polygon_iou(region["polygon_xy"], z["polygon"], (w, h)), 3)})
        if z["confirmed"] and res["prior_zone_consistency"][-1]["iou"] >= rcfg["prior_match_iou"]:
            res["validated_loading_region"] = {**region, "verification_status": "verified",
                                               "validated_by": f"confirmed prior zone {z['id']}"}
    return res


def manual_loading_region(demo_inputs, truck_id):
    """IMAGE_DEMO: region typed in by a person (labelled manual_unconfirmed)."""
    m = demo_inputs.get("loading_region")
    if not m:
        return {"status": "SEMANTIC_REGION_UNCERTAIN", "reason": "IMAGE_DEMO input has no loading_region",
                "truck_object": truck_id, "loading_area_candidates": [], "selection": None,
                "selected_loading_region": None, "validated_loading_region": None, "prior_zone_consistency": []}
    region = validate({"region_id": "loading_area_manual_1", "region_type": "loading_area",
                       "polygon_xy": m["polygon_xy"], "coordinate_frame": RAW_PX,
                       "reference_objects": [truck_id] if truck_id else [], "evidence": [m.get("provenance", "manual")],
                       "verification_status": "manual_unconfirmed", "source": "manual_test_data"}, REGION_SCHEMA)
    return {"status": "MANUAL_REGION_IMAGE_DEMO", "truck_object": truck_id, "loading_area_candidates": [],
            "selection": {"method": "manual_test_data"}, "selected_loading_region": region,
            "validated_loading_region": None, "prior_zone_consistency": []}
