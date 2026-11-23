"""Semantic destination (v0.2): truck -> loading-area candidates -> per-candidate VLM judgement.

Stages kept separate on purpose (failure analysis: docs/ANALYSIS_v0.1_destination_failure.md):
  loading_area_candidates  geometric bands around the truck box (v0.1 baseline)
                           + workspace structures around the truck localised by Qwen3-VL grounding
                             (accepted only after geometric association checks with the truck box)
  semantic judgement       one candidate per image -> yes / no / uncertain. The VLM never supplies the
                           coordinates used downstream and never sees the robot map, reference zones or
                           any accessibility result (v0.1 mixed accessibility into this stage).
  semantic_regions         candidates judged "yes" (VLM inference, unverified)
Placement feasibility (semantics.placement), robot approach and goal (navigation.goal_selector) and path
planning are later stages with their own failure codes.
"""
import cv2
import numpy as np

from ..contracts import RAW_PX, REGION_SCHEMA, validate
from ..geometry.coordinate_frames import box_area, box_gap, containment, iou
from ..perception.grounding import qwen_ground_boxes, qwen_json

JUDGE_SCHEMA = {
    "type": "object", "required": ["answer"],
    "properties": {"answer": {"type": "string", "enum": ["yes", "no", "uncertain"]},
                   "reason": {"type": ["string", "null"]}},
}


def _rect(b):
    return [[round(b[0], 1), round(b[1], 1)], [round(b[2], 1), round(b[1], 1)],
            [round(b[2], 1), round(b[3], 1)], [round(b[0], 1), round(b[3], 1)]]


def band_depth(truck_box, rcfg):
    return max(rcfg["min_band_px"], rcfg["band_ratio"] * max(truck_box[2] - truck_box[0], truck_box[3] - truck_box[1]))


def generate_candidates(truck_box, image_size, rcfg):
    """v0.1 baseline: four axis-aligned bands adjacent to the truck box."""
    w, h = image_size
    x1, y1, x2, y2 = truck_box
    d = band_depth(truck_box, rcfg)
    bands = {"below": [x1, y2, x2, y2 + d], "above": [x1, y1 - d, x2, y1],
             "left": [x1 - d, y1, x1, y2], "right": [x2, y1, x2 + d, y2]}
    out = []
    for side, (a, b, c, e) in bands.items():
        a, b, c, e = max(0, a), max(0, b), min(w, c), min(h, e)
        if (c - a) * (e - b) < rcfg["min_candidate_area_px"]:
            continue
        out.append({"candidate_id": len(out) + 1, "side_of_truck_box": side, "source": f"geometric_band:{side}",
                    "polygon_xy": _rect([a, b, c, e]), "coordinate_frame": RAW_PX,
                    "generation": f"axis-aligned band of depth {round(d, 1)} px adjacent to the truck box"})
    return out


def structure_candidates(client, mcfg, pcfg, image_path, image_size, truck_box):
    """Workspace structures (e.g. a painted bay) localised by Qwen3-VL grounding. A box becomes a
    candidate only if it is associated with the truck (contains it, or lies within one band depth),
    is not scene-sized and is not just the truck box itself."""
    dcfg = pcfg["destination"]
    d = band_depth(truck_box, pcfg["region_grounding"])
    cands, records = [], []
    for q in dcfg["structure_queries"]:
        boxes, rec = qwen_ground_boxes(client, mcfg, image_path, image_size, q)
        r = {"query": q, "call": rec, "boxes": []}
        for b in boxes:
            bb = b["bbox_xyxy"]
            chk = {"truck_containment": round(containment(truck_box, bb), 3), "gap_to_truck_px": round(box_gap(bb, truck_box), 1),
                   "area_ratio_to_truck": round(box_area(bb) / max(1.0, box_area(truck_box)), 3),
                   "iou_with_truck": round(iou(bb, truck_box), 3)}
            reason = None
            if chk["truck_containment"] < dcfg["structure_min_truck_containment"] and chk["gap_to_truck_px"] > d:
                reason = "not associated with the truck (neither contains it nor adjacent)"
            elif chk["area_ratio_to_truck"] > dcfg["structure_max_area_ratio"]:
                reason = "scene-sized box"
            elif chk["iou_with_truck"] > dcfg["structure_max_truck_iou"]:
                reason = "same as the truck box (the vehicle, not an area)"
            elif any(iou(bb, [c["polygon_xy"][0][0], c["polygon_xy"][0][1], c["polygon_xy"][2][0], c["polygon_xy"][2][1]])
                     >= dcfg["structure_dedup_iou"] for c in cands):
                reason = "duplicate of an earlier structure candidate"
            r["boxes"].append({"bbox_xyxy": [round(v, 1) for v in bb], "bbox_norm1000": b["bbox_norm1000"],
                               "association": chk, "accepted": reason is None, "reject_reason": reason})
            if reason is None:
                cands.append({"source": f"grounded_structure:{q}", "side_of_truck_box": "structure",
                              "polygon_xy": _rect(bb), "coordinate_frame": RAW_PX, "association": chk,
                              "generation": f"Qwen3-VL grounding box for '{q}' (0-1000 -> raw px), checked against "
                                            "the truck box"})
        records.append(r)
    return cands, records


def render_judge_input(image, truck_box, polygon, margin):
    h, w = image.shape[:2]
    pts = np.array(polygon + [[truck_box[0], truck_box[1]], [truck_box[2], truck_box[3]]], dtype=np.float64)
    x0, y0 = int(max(0, pts[:, 0].min() - margin)), int(max(0, pts[:, 1].min() - margin))
    x1, y1 = int(min(w, pts[:, 0].max() + margin)), int(min(h, pts[:, 1].max() + margin))
    crop = image[y0:y1, x0:x1].copy()
    tb = [int(truck_box[0] - x0), int(truck_box[1] - y0), int(truck_box[2] - x0), int(truck_box[3] - y0)]
    cv2.rectangle(crop, tb[:2], tb[2:], (255, 255, 255), 2)
    cv2.putText(crop, "T", (tb[0] + 4, tb[1] + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    poly = (np.array(polygon) - [x0, y0]).round().astype(np.int32)
    cv2.polylines(crop, [poly], True, (0, 255, 0), 3)
    scale = max(1.0, 896 / max(crop.shape[:2]))
    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return crop, {"crop_xyxy_raw": [x0, y0, x1, y1], "scale": round(scale, 4),
                  "markers": "white box T = truck, green outline = candidate"}


def judge_candidates(client, mcfg, image, truck_box, cands, out_dir, margin):
    prompt = mcfg["prompts"]["region_judge"]["text"]
    for c in cands:
        crop, info = render_judge_input(image, truck_box, c["polygon_xy"], margin)
        path = out_dir / f"region_judge_cand{c['candidate_id']}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), crop)
        obj, rec = qwen_json(client, mcfg, "region_judge", prompt, path, mcfg["qwen"]["max_new_tokens"]["verify"],
                             JUDGE_SCHEMA)
        c["judgement"] = {"answer": obj["answer"] if obj else "invalid_response",
                          "reason": obj.get("reason") if obj else rec["error"], "input_image": str(path),
                          "input_crop": info, "call": rec}
        c["vlm_plausible"], c["vlm_reason"] = c["judgement"]["answer"], c["judgement"]["reason"]


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


def ground_loading_region(client, mcfg, pcfg, image, image_path, truck, out_dir, prior_zones=()):
    dcfg = pcfg["destination"]
    h, w = image.shape[:2]
    tb = truck["bbox_xyxy"]
    cands, struct_records = [], []
    if "geometric_bands" in dcfg["candidate_sources"]:
        cands += generate_candidates(tb, (w, h), pcfg["region_grounding"])
    if "grounded_structure" in dcfg["candidate_sources"]:
        sc, struct_records = structure_candidates(client, mcfg, pcfg, image_path, (w, h), tb)
        cands += sc
    for i, c in enumerate(cands, 1):
        c["candidate_id"] = i
    res = {"method": "v0.2: candidates from " + "+".join(dcfg["candidate_sources"]) + "; per-candidate VLM judgement",
           "truck_object": truck["object_id"], "loading_area_candidates": cands, "structure_grounding": struct_records,
           "semantic_regions": [], "selected_loading_region": None, "validated_loading_region": None,
           "prior_zone_consistency": [],
           "note": "semantic stage only; placement, approach and path are evaluated later (destination chain)"}
    if not cands:
        res.update(status="NO_LOADING_REGION_CANDIDATES", reason="no candidate area could be generated around the truck")
        return res
    judge_candidates(client, mcfg, image, tb, cands, out_dir, dcfg["judge_crop_margin_px"])
    yes = [c for c in cands if c["vlm_plausible"] == "yes"]
    res["semantic_regions"] = [_region(c, truck["object_id"], [c["vlm_reason"]] if c.get("vlm_reason") else [],
                                       "unverified", c["source"] + "+vlm_judgement") for c in yes]
    summary = {c["candidate_id"]: c["vlm_plausible"] for c in cands}
    if len(yes) == 1:
        res.update(status="SEMANTIC_REGION_FOUND", selected_loading_region=res["semantic_regions"][0])
    elif yes:
        res.update(status="SEMANTIC_REGIONS_MULTIPLE",
                   reason=f"{len(yes)} candidates judged plausible {summary}; choice deferred to placement/approach/path cost")
    elif any(v in ("uncertain", "invalid_response") for v in summary.values()):
        res.update(status="SEMANTIC_REGION_UNCERTAIN", reason=f"no candidate judged 'yes' {summary}")
    else:
        res.update(status="NO_LOADING_REGION_FOUND", reason=f"truck present but every candidate judged 'no' {summary}")
    for reg in res["semantic_regions"]:
        for z in prior_zones:  # evaluation/consistency only; never feeds planning or the model
            row = {"region_id": reg["region_id"], "zone_id": z["id"], "role": z["role"], "confirmed": z["confirmed"],
                   "iou": round(polygon_iou(reg["polygon_xy"], z["polygon"], (w, h)), 3)}
            res["prior_zone_consistency"].append(row)
            if z["confirmed"] and row["iou"] >= pcfg["region_grounding"]["prior_match_iou"]:
                res["validated_loading_region"] = {**reg, "verification_status": "verified",
                                                   "validated_by": f"confirmed prior zone {z['id']}"}
    return res


def manual_loading_region(demo_inputs, truck_id):
    """IMAGE_DEMO: region typed in by a person (labelled manual_unconfirmed)."""
    m = demo_inputs.get("loading_region")
    base = {"truck_object": truck_id, "loading_area_candidates": [], "selection": None, "semantic_regions": [],
            "selected_loading_region": None, "validated_loading_region": None, "prior_zone_consistency": []}
    if not m:
        return {**base, "status": "SEMANTIC_REGION_UNCERTAIN", "reason": "IMAGE_DEMO input has no loading_region"}
    region = validate({"region_id": "loading_area_manual_1", "region_type": "loading_area",
                       "polygon_xy": m["polygon_xy"], "coordinate_frame": RAW_PX,
                       "reference_objects": [truck_id] if truck_id else [], "evidence": [m.get("provenance", "manual")],
                       "verification_status": "manual_unconfirmed", "source": "manual_test_data"}, REGION_SCHEMA)
    return {**base, "status": "MANUAL_REGION_IMAGE_DEMO", "selection": {"method": "manual_test_data"},
            "semantic_regions": [region], "selected_loading_region": region}
