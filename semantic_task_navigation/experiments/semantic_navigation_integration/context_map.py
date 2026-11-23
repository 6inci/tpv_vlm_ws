"""Experiment 1: task-oriented semantic context map (nodes, typed relations, provenance). Pure functions; no model call.

Only task-relevant entities become nodes (inclusion_rule). A relation is created ONLY when its evidence exists:
  provenance INSTRUCTION_GIVEN  = stated by the user's command (never checked against the image here -> status UNVERIFIED)
             VLM_INFERRED       = stated by the STORED Qwen reply (-> UNVERIFIED)
             GEOMETRY_DERIVED   = computed in Python from detector boxes / floor polygons (-> SUPPORTED on its stated basis)
Image-space thresholds are the v0.2 ones (config/planning.yaml relations); tall objects get image-space relations only.
"""
import cv2
import numpy as np

from semantic_task_nav.geometry.coordinate_frames import box_gap, containment

PROVENANCE = ("INSTRUCTION_GIVEN", "VLM_INFERRED", "GEOMETRY_DERIVED")
STATUSES = ("SUPPORTED", "UNVERIFIED", "UNKNOWN")
INCLUSION_RULE = ["the instruction's target class (all retained red-cube candidates) and its source class (conveyor)",
                  "the robot", "destination candidates (stored Qwen functional areas) and their grounding (painted bay, its free floor, the truck)",
                  "map layers that bound motion (DRAFT zones present in the E1 map)", "verified obstacles next to the destination (cones, Qwen crop 'yes')"]


def box_poly(b):
    return [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]


class Raster:
    """Polygon masks on one canvas: image px (scale 1, offset 0) or floor cm (scale = px/cm, offset = pad cm)."""

    def __init__(self, size_wh, scale=1.0, offset=0.0):
        self.w, self.h, self.s, self.o = int(size_wh[0]), int(size_wh[1]), float(scale), float(offset)

    def mask(self, polys):
        m = np.zeros((self.h, self.w), np.uint8)
        for p in polys:
            cv2.fillPoly(m, [np.round((np.asarray(p, np.float64) + self.o) * self.s).astype(np.int32)], 1)
        return m.astype(bool)

    def frac_inside(self, inner, outer):
        a = self.mask([inner])
        n = int(a.sum())
        return float((a & self.mask([outer])).sum() / n) if n else 0.0

    def gap(self, a, b):
        """Smallest distance between two polygons in canvas units (0 if they touch or overlap)."""
        ma, mb = self.mask([a]), self.mask([b])
        if not ma.any() or not mb.any():
            return None
        dt = cv2.distanceTransform((~mb).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        return float(dt[ma].min() / self.s)


def source_relation(cand_box, src_box, near_px, contain_ratio):
    """'on' (candidate box inside the source box, 2D), 'near' (gap <= near_px) or 'none'. Image space only."""
    c, gap = containment(cand_box, src_box), box_gap(cand_box, src_box)
    kind = "on" if c >= contain_ratio else "near" if gap <= near_px else "none"
    return kind, {"containment_in_source_box": round(c, 3), "gap_px": round(gap, 1), "contain_ratio": contain_ratio, "near_px": near_px}


def build(inp, ccfg, near_px, contain_ratio):
    """inp: see run_integration.collect_context_inputs. Returns the map dict (JSON-serialisable)."""
    img_r = Raster(inp["image_size"])
    flo_r = Raster(inp["floor_canvas_wh"], inp["floor_px_per_cm"], inp["floor_pad_cm"])
    g = {"frame": {"image": "undistorted E1 px (1920x1080)", "floor": "floor_cm via the PROVISIONAL mat-seam homography (status PROVISIONAL_UNVALIDATED)",
                   "note": "image boxes of tall objects are not floor footprints; only floor-level entities get floor polygons"},
         "inclusion_rule": INCLUSION_RULE, "nodes": [], "relations": [], "vlm_mentions": [], "conflicts": [], "excluded": [], "unresolved": []}
    ids = set()

    def node(**kw):
        assert kw["id"] not in ids, kw["id"]
        ids.add(kw["id"])
        g["nodes"].append(kw)
        return kw["id"]

    def rel(rtype, subj, obj, prov, status, basis, evidence):
        assert prov in PROVENANCE and status in STATUSES and subj in ids and obj in ids
        if any(r["type"] == rtype and r["subject"] == subj and r["object"] == obj for r in g["relations"]):
            return                                         # one relation per (type, subject, object); the first basis is kept
        assert not (prov != "GEOMETRY_DERIVED" and status == "SUPPORTED"), "instruction/VLM statements are not checked here"
        g["relations"].append({"id": f"r{len(g['relations']) + 1}", "type": rtype, "subject": subj, "object": obj, "provenance": prov,
                               "status": status, "basis": basis, "evidence": evidence})

    # ---- task (instruction)
    rp = inp["rule_parse"]
    node(id="task", kind="task", command=inp["command"],
         target={"value": rp["target"], "provenance": "INSTRUCTION_GIVEN", "parser": "v0.2 rule_based_parse (deterministic)"},
         destination={"value": None, "provenance": "INSTRUCTION_GIVEN", "note": "the command names a purpose ('출하할 수 있는 적절한 장소'), no place; "
                       "the rule parser returns missing_information. The destination comes from the stored Qwen reply (VLM_INFERRED)."})

    # ---- robot, objects
    rb = inp["robot"]
    node(id="robot", kind="robot", bbox_xyxy=rb["bbox_xyxy"], start_cm=rb["start_cm"], heading="UNKNOWN",
         evidence=[{"kind": "DETECTOR_OBSERVED", "detail": "Grounding DINO 'mobile robot.' (stored)"}, {"kind": "VLM_CROP_VERIFIED", "detail": f"stored Qwen crop check: {rb['crop_answer']}"}],
         status="ESTIMATED floor start (box centre); heading UNKNOWN", floor_body_fits_at_start=rb["body_fits_at_start"])
    for o in inp["objects"]:
        node(id=o["object_id"], kind="object", class_name=o["class_name"], bbox_xyxy=[round(v, 1) for v in o["bbox_xyxy"]],
             floor_cm=o.get("floor_point_cm"), floor_note=o.get("floor_note"),
             evidence=[{"kind": "DETECTOR_OBSERVED", "detail": o["source"]}, {"kind": "VLM_CROP_VERIFIED", "detail": o["crop"]}],
             role={"conveyor": "instruction source object; support-surface candidate of the target", "truck": "destination context",
                   "traffic_cone": "obstacle next to the destination"}.get(o["class_name"]))
    # ---- painted bay and its free floor (Python, stored)
    bay = inp["bay"]
    node(id="loading_bay", kind="floor_region", polygon_xy=bay["polygon_px"], polygon_cm=bay["polygon_cm"], status=bay["status"],
         evidence=[{"kind": "GEOMETRY_DERIVED", "detail": "stored dashed-line localization (Python, semantic_evidence_verification)"}], not_verified=bay["not_verified"])
    pf = inp["placement_floor"]
    node(id="placement_floor", kind="floor_region", polygons_xy=pf["polygons_px"], polygons_cm=pf["polygons_cm"], status="GEOMETRY_ONLY (not confirmed empty)",
         evidence=[{"kind": "GEOMETRY_DERIVED", "detail": pf["basis"]}], definition="empty floor inside the dashed bay (user's provisional placement definition)")
    # ---- stored Qwen functional areas
    qs = inp["qwen_structured"]
    chosen = qs["task_analysis"]["destination_area_id"]
    for a in qs["functional_areas"]:
        if a["id"] in ccfg["excluded_vlm_areas"]:
            g["excluded"].append({"id": a["id"], "function_name": a["function_name"], "reason": ccfg["excluded_vlm_areas"][a["id"]]})
            continue
        node(id=a["id"], kind="vlm_functional_area", function_name=a["function_name"], bbox_xyxy=[round(v, 1) for v in inp["area_boxes_px"][a["id"]]],
             floor_projection_cm=inp["area_floor_cm"][a["id"]], vlm_confidence=a.get("confidence"), observed_evidence=a["observed_evidence"], inference=a["inference"],
             evidence=[{"kind": "VLM_INFERRED", "detail": f"stored Qwen L2 reply (run {inp['qwen_meta']['run']}, for the wording '{inp['qwen_meta']['instruction']}')"}],
             chosen_by_vlm=a["id"] == chosen)
    # ---- DRAFT zones
    for z in inp["zones"]:
        node(id=z["id"], kind="map_zone", layer=z["layer"], name_ko=z.get("name_ko"), role=z.get("role"), access_draft=z.get("access_draft"),
             polygon_xy=z["polygon_px"], polygon_cm=z["polygon_cm"], evidence=[{"kind": "MANUAL_MAP_PRIOR", "detail": "hand-drawn DRAFT zone, transferred to E1 (DRAFT_UNCONFIRMED)"}])
    # ---- target candidates
    for c in inp["candidates"]:
        node(id=c["id"], kind="target_candidate", rank=c["rank"], bbox_xyxy=c["bbox_xyxy"], source_types=c["source_types"], detector_score=c["score"],
             state=c["state"], stored_qwen=c["stored_qwen"], crop_views=c["crop_views"], floor_projection_cm=c["floor_cm"],
             floor_note="floor-plane projection of the image box; the cube height above the floor is UNKNOWN (parallax)",
             evidence=[{"kind": "DETECTOR_OBSERVED", "detail": ",".join(c["source_types"])}] + ([{"kind": "VLM_CROP_VERIFIED", "detail": f"stored view A: {c['stored_qwen']['status']}"}] if c["stored_qwen"] else []))

    # ================================================================ relations
    src_cls = (rp.get("target") or {}).get("source_reference")
    srcs = [o for o in inp["objects"] if o["class_name"] == src_cls]
    if src_cls and len(srcs) == 1:
        s = srcs[0]
        rel("source_referenced_by_instruction", "task", s["object_id"], "INSTRUCTION_GIVEN", "UNVERIFIED", "instruction_text",
            {"clause": "컨베이어에서 생산된", "parsed": {"source_reference": src_cls, "source_relation": rp["target"]["source_relation"]},
             "grounding": f"class match to the only accepted {src_cls} ({s['object_id']}: detector + Qwen crop 'yes')",
             "why_unverified": "a still image cannot show that the object was produced by this conveyor"})
        for c in inp["candidates"]:
            kind, ev = source_relation(c["bbox_xyxy"], s["bbox_xyxy"], near_px, contain_ratio)
            if kind != "none":
                rel(kind, c["id"], s["object_id"], "GEOMETRY_DERIVED", "SUPPORTED", "image_2d",
                    {**ev, "limit": "2D image boxes; 3D support ('on' the belt surface) is NOT established"})
    elif src_cls:
        g["unresolved"].append(f"instruction refers to a {src_cls}; {len(srcs)} accepted {src_cls} objects -> no source grounding")
    rel("candidate_destination_for", chosen, "task", "VLM_INFERRED", "UNVERIFIED", "stored_vlm_reply",
        {"decision": qs["task_analysis"]["decision"], "reason": qs["task_analysis"]["reason"], "place_needed_for_task": qs["task_analysis"]["place_needed_for_task"],
         "info_added_by_vlm": qs["task_analysis"]["info_added_by_you"], "stored_for_wording": inp["qwen_meta"]["instruction"]})
    # VLM area <-> grounded entities (image 2D): fraction of the entity inside the area box >= contain_ratio
    grounded = [("robot", box_poly(rb["bbox_xyxy"]))] + [(o["object_id"], box_poly(o["bbox_xyxy"])) for o in inp["objects"]] + \
               [("loading_bay", bay["polygon_px"])] + [("placement_floor", p) for p in pf["polygons_px"][:1]]
    for a in [n for n in g["nodes"] if n["kind"] == "vlm_functional_area"]:
        for nid, poly in grounded:
            f = img_r.frac_inside(poly, box_poly(a["bbox_xyxy"]))
            if f >= contain_ratio:
                rel("associated_with", a["id"], nid, "GEOMETRY_DERIVED", "SUPPORTED", "image_2d",
                    {"fraction_of_entity_inside_area_box": round(f, 3), "contain_ratio": contain_ratio,
                     "limit": "locates the VLM area on observed entities; the area's FUNCTION stays VLM_INFERRED"})
    tk = bay["stored_truck_check"]
    if tk and tk["status"] == "PASS":
        rel("associated_with", "truck_1", "loading_bay", "GEOMETRY_DERIVED", "SUPPORTED", "image_2d",
            {"stored_check": tk["check"], "value": tk["detail"], "source": "stored bay localization (declared threshold 0.9)"})
    rel("inside", "placement_floor", "loading_bay", "GEOMETRY_DERIVED", "SUPPORTED", "image_2d", {"by_construction": pf["basis"]})
    for o in inp["objects"]:
        if o["class_name"] == "traffic_cone":
            gp = img_r.gap(box_poly(o["bbox_xyxy"]), bay["polygon_px"])
            if gp is not None and gp <= near_px:
                rel("near", o["object_id"], "loading_bay", "GEOMETRY_DERIVED", "SUPPORTED", "image_2d", {"gap_px": round(gp, 1), "near_px": near_px})
    # floor-level relations (PROVISIONAL frame)
    floor_ents = [(z["id"], z["polygon_cm"]) for z in inp["zones"]] + [("loading_bay", bay["polygon_cm"])] + [("placement_floor", p) for p in pf["polygons_cm"][:1]]
    for i, (a, pa) in enumerate(floor_ents):
        for b, pb in floor_ents[i + 1:]:
            fa, fb = flo_r.frac_inside(pa, pb), flo_r.frac_inside(pb, pa)
            if max(fa, fb) >= contain_ratio:
                s_, o_ = (a, b) if fa >= fb else (b, a)
                rel("inside", s_, o_, "GEOMETRY_DERIVED", "SUPPORTED", "floor_cm_provisional", {"fraction_inside": round(max(fa, fb), 3), "contain_ratio": contain_ratio})
                continue
            gp = flo_r.gap(pa, pb)
            if gp is not None and gp <= ccfg["floor_adjacent_cm"]:
                rel("adjacent_to", a, b, "GEOMETRY_DERIVED", "SUPPORTED", "floor_cm_provisional", {"gap_cm": round(gp, 2), "adjacent_cm": ccfg["floor_adjacent_cm"]})
    drv = [z for z in inp["zones"] if z["layer"] == "traversable"]
    for z in drv:
        if rb["body_fits_at_start"] and cv2.pointPolygonTest(np.float32(z["polygon_cm"]), tuple(map(float, rb["start_cm"])), False) >= 0:
            rel("inside", "robot", z["id"], "GEOMETRY_DERIVED", "SUPPORTED", "floor_cm_provisional",
                {"start_cm": rb["start_cm"], "body": "35x24 cm + margin fits on free floor at the start (fixed heading 0)", "start_status": "ESTIMATED"})
        bay_gap = flo_r.gap(bay["polygon_cm"], z["polygon_cm"])
        next(n for n in g["nodes"] if n["id"] == "loading_bay")[f"floor_gap_to_{z['id']}_cm"] = None if bay_gap is None else round(bay_gap, 1)
    for c in inp["candidates"]:
        ctr = tuple(map(float, np.mean(c["floor_cm"], axis=0)))
        for z in inp["zones"]:
            if cv2.pointPolygonTest(np.float32(z["polygon_cm"]), ctr, False) >= 0:
                rel("inside", c["id"], z["id"], "GEOMETRY_DERIVED", "SUPPORTED", "floor_cm_provisional",
                    {"floor_projection_centre_cm": [round(v, 1) for v in ctr], "limit": "projection of an object of UNKNOWN height: displaced by parallax"})
    # ---- unlocalised VLM mentions and conflicts
    kw = {"conveyor": [o["object_id"] for o in srcs], "truck": ["truck_1"], "cone": [o["object_id"] for o in inp["objects"] if o["class_name"] == "traffic_cone"],
          "robotic vehicle": ["robot"]}
    for m in qs["observed_objects"]:
        hits = [n for k, v in kw.items() if k in m["label"].lower() for n in v]
        g["vlm_mentions"].append({"label": m["label"], "count": m.get("count"), "location_text": m.get("location_in_image"), "matched_nodes": hits,
                                  "provenance": "VLM_INFERRED", "note": "no box in the reply; kept as text" + ("" if hits else " (not task-relevant / not grounded)")})
    for st in qs["task_analysis"]["info_added_by_you"]:
        if "cube" in st["statement"].lower() and "not visible" in st["statement"].lower():
            g["conflicts"].append({"vlm_statement": st["statement"], "basis": st["basis"], "provenance": "VLM_INFERRED",
                                   "versus": f"{len(inp['candidates'])} detector candidates of the target class ({sum(c['state'] == 'PROBABLE' for c in inp['candidates'])} PROBABLE)",
                                   "handling": "kept as is; neither side is treated as ground truth"})
    cones_vlm = next((m.get("count") for m in qs["observed_objects"] if "cone" in m["label"].lower()), None)
    if cones_vlm is not None:
        g["conflicts"].append({"vlm_statement": f"{cones_vlm} cones", "versus": f"{len(kw['cone'])} cone boxes verified by detector + Qwen crop", "handling": "count not used"})
    g["summary"] = {"n_nodes": len(g["nodes"]), "nodes_by_kind": _count(g["nodes"], "kind"), "n_relations": len(g["relations"]),
                    "relations_by_type": _count(g["relations"], "type"), "relations_by_provenance": _count(g["relations"], "provenance"),
                    "relations_by_status": _count(g["relations"], "status")}
    return g


def _count(items, key):
    out = {}
    for x in items:
        out[x[key]] = out.get(x[key], 0) + 1
    return out


def source_level(cmap, cand_id):
    """The relation the decision module reads: 'on' / 'near' / 'none' between a candidate and the instruction's source object."""
    src = next((r["object"] for r in cmap["relations"] if r["type"] == "source_referenced_by_instruction"), None)
    if src is None:
        return "none", None
    r = next((r for r in cmap["relations"] if r["subject"] == cand_id and r["object"] == src and r["type"] in ("on", "near")), None)
    return (r["type"], r["id"]) if r else ("none", None)
