"""Verification of a VLM-proposed semantic destination (STEP 4/5). Pure geometry + bookkeeping, image space (raw px).

Every check returns a record with a status:
  VERIFIED      established by something other than the VLM's own claim (geometry, detector + crop verification, map prior)
  ESTIMATED     a judgement by the same VLM family (v0.2 region_judge): useful as a filter, not independent evidence
  NOT_CONFIRMED a cited object that the detectors could not confirm (hallucination OR detector miss; not blocking)
  UNVERIFIABLE  no existing tool can check it (e.g. floor lines); listed, never counted as verified
  FAILED        checked and contradicted
  NOT_RUN       an earlier required check failed, or the tool was not needed
`required` checks gate the hand-over to the v0.2 placement / approach / path chain. Nothing here uses the human
reference zone (Z06) or any human-drawn answer region; the map layers are used only AFTER the proposal exists, as an
independent exclusion check, and only when v0.2 says the map is ALIGNED with the image.
"""
from semantic_task_nav.geometry.coordinate_frames import box_area, box_gap, containment, iou
from semantic_task_nav.semantics.region_grounding import band_depth

VERIFIED, ESTIMATED, NOT_CONFIRMED = "VERIFIED", "ESTIMATED", "NOT_CONFIRMED"
UNVERIFIABLE, FAILED, NOT_RUN = "UNVERIFIABLE", "FAILED", "NOT_RUN"


def check(cid, name, status, basis, detail, required=True):
    return {"id": cid, "name": name, "status": status, "required": required, "basis": basis, "detail": detail}


def norm_to_px(b, w, h):
    return [b[0] / 1000 * w, b[1] / 1000 * h, b[2] / 1000 * w, b[3] / 1000 * h]


def rect(b):
    return [[round(b[0], 1), round(b[1], 1)], [round(b[2], 1), round(b[1], 1)],
            [round(b[2], 1), round(b[3], 1)], [round(b[0], 1), round(b[3], 1)]]


def inter_box(a, b):
    box = [max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])]
    return box if box[2] > box[0] and box[3] > box[1] else None


def destination_of(structured, size):
    """(area dict, box_px) of the VLM's chosen destination, or (None, None) when it withheld / names no valid area."""
    ta = structured["task_analysis"]
    if ta["decision"] != "proposed":
        return None, None
    a = next((a for a in structured["functional_areas"] if a["id"] == ta["destination_area_id"]), None)
    return (a, norm_to_px(a["bbox_2d"], *size)) if a else (None, None)


# ------------------------------------------------------------------ V1 image link
def image_link(box, size, v):
    w, h = size
    frac = box_area(box) / (w * h)
    sides = (box[2] - box[0], box[3] - box[1])
    inside = box[0] >= 0 and box[1] >= 0 and box[2] <= w + 1 and box[3] <= h + 1
    ok = inside and min(sides) >= v["min_box_side_px"] and frac <= v["max_area_frac"]
    return check("V1", "proposal is a valid, non-scene-sized region in this image",
                 VERIFIED if ok else FAILED, "python_geometry",
                 {"box_px": [round(x, 1) for x in box], "area_frac_of_image": round(frac, 4),
                  "max_area_frac": v["max_area_frac"], "min_side_px": round(min(sides), 1),
                  "inside_image": bool(inside)})


# ------------------------------------------------------------------ V2 cited evidence
def cited_items(area, vocab):
    """Which vocabulary classes does the VLM cite for its destination, and which evidence strings match no class."""
    texts = [area["function_name"], area["inference"]] + list(area["observed_evidence"])
    low = " ".join(texts).lower()
    cited = [{"class": k, "terms_found": [t for t in spec["terms"] if t in low], "anchor": bool(spec.get("anchor")),
              "source": spec["source"]} for k, spec in vocab.items() if any(t in low for t in spec["terms"])]
    all_terms = [t for spec in vocab.values() for t in spec["terms"]]
    unverifiable = [e for e in area["observed_evidence"] if not any(t in e.lower() for t in all_terms)]
    return cited, unverifiable


def evidence_existence(cited, unverifiable, accepted, cone_hits):
    """accepted: v0.2 accepted detections. cone_hits: {class: [verified boxes]} from the ad-hoc detector path."""
    rows, blocking = [], False
    for c in cited:
        if c["source"] == "v02_detection":
            objs = [o["object_id"] for o in accepted if o["class_name"] == c["class"]]
            present = bool(objs)
            rows.append({**c, "found": objs, "status": VERIFIED if present else (FAILED if c["anchor"] else NOT_CONFIRMED)})
        else:
            hits = cone_hits.get(c["class"])
            rows.append({**c, "found": None if hits is None else len(hits),
                         "status": NOT_RUN if hits is None else (VERIFIED if hits else NOT_CONFIRMED)})
        blocking |= c["anchor"] and rows[-1]["status"] != VERIFIED
    has_anchor = any(c["anchor"] for c in cited)
    status = FAILED if (blocking or not has_anchor) else VERIFIED
    return check("V2", "the objects cited as evidence exist in the image (anchor object required)", status,
                 "detector+vlm_crop_verification (v0.2 detections / ad-hoc DINO)",
                 {"cited": rows, "anchor_cited": has_anchor, "unverifiable_evidence": unverifiable,
                  "note": "an anchor (truck) must be cited AND verified; uncited overlap with the truck is not evidence"})


# ------------------------------------------------------------------ V3 anchor association (v0.2 thresholds)
def association(box, truck_box, dcfg, rcfg):
    d = band_depth(truck_box, rcfg)
    m = {"truck_containment": round(containment(truck_box, box), 3), "gap_to_truck_px": round(box_gap(box, truck_box), 1),
         "area_ratio_to_truck": round(box_area(box) / max(1.0, box_area(truck_box)), 3),
         "iou_with_truck": round(iou(box, truck_box), 3), "association_distance_px": round(d, 1)}
    why = None
    if m["truck_containment"] < dcfg["structure_min_truck_containment"] and m["gap_to_truck_px"] > d:
        why = "not associated with the verified truck (neither contains it nor lies within one band depth)"
    elif m["area_ratio_to_truck"] > dcfg["structure_max_area_ratio"]:
        why = "scene-sized relative to the truck"
    elif m["iou_with_truck"] > dcfg["structure_max_truck_iou"]:
        why = "same as the truck box (the vehicle itself, not an area)"
    return check("V3", "region is geometrically associated with the verified anchor (v0.2 association rules)",
                 FAILED if why else VERIFIED, "python_geometry (v0.2 structure_* thresholds)", {**m, "reject_reason": why})


def association_zone(truck_box, size, rcfg):
    """Truck box dilated by v0.2's own band depth: the only part of a proposal that a verified anchor supports."""
    d, (w, h) = band_depth(truck_box, rcfg), size
    return [max(0.0, truck_box[0] - d), max(0.0, truck_box[1] - d), min(float(w), truck_box[2] + d), min(float(h), truck_box[3] + d)]


# ------------------------------------------------------------------ refinement by grounding the VLM's own evidence
def screen_refined(box, vlm_box, truck_box, dcfg, rcfg, refine):
    """A box returned by grounding one of the VLM's own phrases: usable only if it is inside the VLM box and passes
    the same association rules. Returns (ok, reason)."""
    inside = containment(box, vlm_box)
    if inside < refine["min_inside_vlm_box"]:
        return False, f"only {inside:.2f} of the box lies inside the VLM proposal"
    a = association(box, truck_box, dcfg, rcfg)
    if a["status"] == FAILED:
        return False, a["detail"]["reject_reason"]
    if containment(truck_box, box) < dcfg["structure_min_truck_containment"]:
        return False, "does not enclose the verified truck (a refined region must surround its anchor)"
    return True, None


# ------------------------------------------------------------------ V5 no inappropriate inclusion
def _frac(mask, box):
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    sub = mask[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
    return float(sub.mean()) if sub.size else 0.0


def inclusion(region_box, others, areas, layers, v, size):
    """others: verified non-anchor detections [{object_id,class_name,bbox_xyxy}]. areas: the VLM's other proposed
    areas [{id,function_name,box_px}]. layers: NavMap.full_layers or None (map not aligned)."""
    problems, notes = [], []
    for o in others:
        c = containment(o["bbox_xyxy"], region_box)
        (problems if c >= v["max_other_object_containment"] else notes).append(
            {"object": o["object_id"], "class": o["class_name"], "fraction_of_object_inside_region": round(c, 3)})
    w, h = size
    for a in areas:
        if box_area(a["box_px"]) / (w * h) > v["scene_sized_other_area_frac"]:
            notes.append({"area": a["id"], "function": a["function_name"], "ignored": "scene-sized (degenerate) proposal"})
            continue
        inter = inter_box(region_box, a["box_px"])
        if not inter:
            continue
        ia = box_area(inter)
        frac = max(ia / max(1.0, box_area(a["box_px"])), ia / max(1.0, box_area(region_box)))
        row = {"area": a["id"], "function": a["function_name"], "overlap": round(frac, 3)}
        conflict = any(t in a["function_name"].lower() for t in v["conflicting_function_terms"])
        if frac >= v["conflicting_area_overlap"] and conflict:
            problems.append({**row, "reason": "overlaps another proposed area with a conflicting function (VLM's own label)"})
        elif frac > 0.05:
            notes.append(row)
    map_detail = {"status": UNVERIFIABLE, "reason": "no aligned map: robot-drivable / no-go zones cannot be checked"}
    if layers is not None:
        drive, nogo = _frac(layers["traversable"], region_box), _frac(layers["no_go"], region_box)
        bad = drive > v["max_drive_zone_overlap_frac"] or nogo > v["max_no_go_overlap_frac"]
        map_detail = {"status": FAILED if bad else VERIFIED, "drive_zone_fraction": round(drive, 4),
                      "no_go_fraction": round(nogo, 4), "basis": "map prior review_status DRAFT_UNCONFIRMED (independent, "
                                                                 "never shown to the VLM)"}
        if bad:
            problems.append({"map": map_detail})
    status = FAILED if problems else (VERIFIED if layers is not None else ESTIMATED)
    return check("V5", "region does not inappropriately contain other equipment, conflicting areas or travel zones", status,
                 "python_geometry + map prior (if aligned)",
                 {"problems": problems, "notes": notes, "map_check": map_detail,
                  "note": "ESTIMATED = image-only checks passed but the map-based check could not be run"})


# ------------------------------------------------------------------ decision
def decide(checks, map_aligned, precision):
    """Gate. Returns (status, reasons). Required checks must be VERIFIED. ESTIMATED is accepted only for V3b (a tight,
    unrefined box) and V4 (same-model judge, used as a filter); V5 may be ESTIMATED only when the map is not aligned,
    which stops planning but keeps the semantic result."""
    reasons = []
    for c in checks:
        if not c["required"]:
            continue
        ok = c["status"] == VERIFIED or (c["id"] in ("V3b", "V4") and c["status"] == ESTIMATED) or \
            (c["id"] == "V5" and c["status"] == ESTIMATED and not map_aligned)
        if not ok:
            reasons.append(f"{c['id']} {c['status']}: {c['name']}")
    if reasons:
        return "WITHHELD", reasons
    return ("ADOPTED" if map_aligned else "ADOPTED_SEMANTIC_ONLY_MAP_NOT_ALIGNED"), [f"spatial precision: {precision}"]
