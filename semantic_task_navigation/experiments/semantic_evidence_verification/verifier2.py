"""V2: evidence- and purpose-based verification of a VLM-proposed semantic destination.

Independent axes (kept apart on purpose; V1 mixed them into one gate):
  A  evidence      each cited object is looked up in what detectors observed (never in the VLM's own wording)
  B  relation      a verified evidence object must stand in a spatial relation to the proposed region
  C  purpose fit   a verified, related object must carry a capability the INSTRUCTION's purpose needs
  ->  semantic verdict (A+B+C)
  D  placement     separate from the semantic verdict; region SIZE is never used. Needs a verified localization
  E  planner       can the existing planning chain represent the destination (obstacle coverage), separate again

Nothing here names a required facility, reads the VLM's function description as evidence, uses the reference zone Z06
or adjusts a threshold: the relation thresholds are v0.2's (config/planning.yaml). Human priors (the evidence registry
and the purposes table) are DATA in config.yaml.

Image space only (raw_image_px). No model is called here; observations are passed in (stored runs or synthetic input).
"""
import re
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from semantic_task_nav.geometry.coordinate_frames import box_area, box_gap, containment  # noqa: E402
from semantic_task_nav.semantics.region_grounding import band_depth  # noqa: E402

def load_config(path=None):
    """config.yaml with the legacy V1 limit resolved from V1's own config (read, not copied)."""
    import yaml
    path = Path(path or Path(__file__).resolve().parent / "config.yaml")
    cfg = yaml.safe_load(open(path, encoding="utf-8"))
    leg = cfg["legacy_v1"]
    v1 = yaml.safe_load(open((path.parent / leg["config"]).resolve(), encoding="utf-8"))
    node = v1
    for k in leg["key"].split("."):
        node = node[k]
    cfg["legacy_v1"]["max_unrefined_area_ratio"] = node
    return cfg


def legacy_size_gate(box, obs, cfg):
    """What V1's size gate would say about this region. Reported next to the V2 verdict; never used by V2."""
    trucks = [o for o in obs["objects"] if o["class_name"] == "truck"]
    limit = cfg["legacy_v1"]["max_unrefined_area_ratio"]
    if len(trucks) != 1:
        return {"limit": limit, "ratio_to_truck": None, "would_block_in_v1": None, "role": "comparison only",
                "note": "V1 measures the region against EXACTLY ONE truck box; here that reference does not exist"}
    ratio = box_area(box) / max(1.0, box_area(trucks[0]["bbox_xyxy"]))
    return {"limit": limit, "ratio_to_truck": round(ratio, 3), "would_block_in_v1": ratio > limit,
            "role": "comparison only (V1 lets a refined box bypass it); V2 does not use region size"}


# ---- claim / class evidence statuses
VERIFIED_PRESENT = "VERIFIED_PRESENT"     # a detector + crop check confirmed at least one instance
NOT_FOUND = "NOT_FOUND"                   # the class WAS searched and no instance was confirmed
NOT_RUN = "NOT_RUN"                       # a detector exists for the class but no observation of it is available
UNVERIFIABLE = "UNVERIFIABLE"             # no detector is registered for this kind of claim
NOT_EVIDENCE = "NOT_EVIDENCE"             # the object being carried is not evidence for where to put it
# ---- relation statuses
HOLDS, FAILS = "RELATION_HOLDS", "RELATION_FAILS"
# ---- semantic verdicts and final dispositions
SEMANTICALLY_SUPPORTED = "SEMANTICALLY_SUPPORTED"
EVIDENCE_NOT_ESTABLISHED = "EVIDENCE_NOT_ESTABLISHED"
EVIDENCE_RELATION_INCONSISTENT = "EVIDENCE_RELATION_INCONSISTENT"
PURPOSE_FIT_UNDETERMINED = "PURPOSE_FIT_UNDETERMINED"
PLACEMENT_UNVERIFIED = "SEMANTICALLY_SUPPORTED_PLACEMENT_UNVERIFIED"
PLANNER_UNSUPPORTED = "SEMANTICALLY_SUPPORTED_PLANNER_UNSUPPORTED"
READY = "SEMANTICALLY_SUPPORTED_READY_FOR_PLANNING"
LABELS_KO = {
    EVIDENCE_NOT_ESTABLISHED: "시각적 근거가 확인되지 않은 후보",
    EVIDENCE_RELATION_INCONSISTENT: "근거와 공간 관계가 불일치하는 후보",
    PURPOSE_FIT_UNDETERMINED: "작업 목적에 적합한지 확정할 수 없는 후보",
    SEMANTICALLY_SUPPORTED: "의미적으로 타당한 후보",
    PLACEMENT_UNVERIFIED: "의미적으로는 타당하지만 배치 가능성이 검증되지 않은 후보",
    PLANNER_UNSUPPORTED: "검증된 의미 후보이나 기존 경로계획 인터페이스에서 지원하지 않는 유형",
    READY: "의미 검증과 배치 근거가 모두 확인된 후보 (계획 입력 가능)",
}


# ------------------------------------------------------------------ purpose (from the INSTRUCTION, not from the VLM)
def purpose_from_instruction(instruction, purposes):
    low = instruction.lower()
    hits = {name: [t for t in p["instruction_terms"] if t.lower() in low] for name, p in purposes.items()}
    hits = {k: v for k, v in hits.items() if v}
    requires = sorted({c for k in hits for c in purposes[k]["requires_any_of"]})
    return {"purposes": sorted(hits), "matched_terms": hits, "requires_any_of": requires,
            "known": bool(hits), "source": "instruction text (keyword table in config.yaml; knowledge: manual_prior)"}


# ------------------------------------------------------------------ A. claims -> registry classes
def _match(term, text):
    return re.search(r"\b" + re.escape(term.lower()), text.lower()) is not None


def extract_claims(observed_evidence, registry, marking_terms):
    """Only `observed_evidence` entries are claims. Each is typed: registry (names a class the detectors know),
    marking (lines/symbols: UNVERIFIABLE, not an obstacle) or unmatched_object (UNVERIFIABLE, a possible obstacle)."""
    claims = []
    for text in observed_evidence:
        classes = [k for k, spec in registry.items() if any(_match(t, text) for t in spec["terms"])]
        kind = "registry" if classes else ("marking" if any(_match(t, text) for t in marking_terms) else "unmatched_object")
        hedged = any(_match(w, text) for w in ("assumed", "likely", "probably", "possibly", "maybe", "seems", "appears"))
        claims.append({"text": text, "classes": classes, "kind": kind,
                       "hedged": hedged})     # recorded only: the VLM itself flags a guess; the verdict does not depend on it
    return claims


def reasoning_only_classes(reasoning_text, claims, registry):
    """Classes the VLM mentions only in its reasoning, not as evidence. Informational: reasoning is not evidence."""
    cited = {c for cl in claims for c in cl["classes"]}
    return sorted(k for k, spec in registry.items() if k not in cited and any(_match(t, reasoning_text) for t in spec["terms"]))


# ------------------------------------------------------------------ B. relation of an observed object to the region
def relation_of(obj_box, region_box, v02cfg):
    min_in = v02cfg["destination"]["structure_min_truck_containment"]
    depth = band_depth(obj_box, v02cfg["region_grounding"])
    inside, gap = containment(obj_box, region_box), box_gap(obj_box, region_box)
    return {"fraction_of_object_inside_region": round(inside, 3), "gap_px": round(gap, 1), "adjacency_distance_px": round(depth, 1),
            "min_inside_fraction": min_in, "relation": HOLDS if (inside >= min_in or gap <= depth) else FAILS,
            "rule": "inside >= v0.2 structure_min_truck_containment OR gap <= band_depth(object) (both from config/planning.yaml)"}


def class_evidence(cls, region_box, obs, registry, v02cfg):
    spec = registry[cls]
    out = {"class": cls, "backend": spec["backend"], "role": spec["role"], "capabilities": spec["capabilities"],
           "obstacle": spec["obstacle"], "instances": [], "n_related": 0}
    if spec["role"] == "task_object":
        return {**out, "status": NOT_EVIDENCE, "note": "the object to be carried says nothing about where it should go"}
    insts = [o for o in obs["objects"] if o["class_name"] == cls]
    for o in insts:
        out["instances"].append({"object_id": o["object_id"], "bbox_xyxy": [round(v, 1) for v in o["bbox_xyxy"]],
                                 "basis": o.get("basis"), **relation_of(o["bbox_xyxy"], region_box, v02cfg)})
    out["n_related"] = sum(1 for i in out["instances"] if i["relation"] == HOLDS)
    searched = obs["searched"].get(cls)
    if insts:
        out["status"] = VERIFIED_PRESENT
    elif searched and searched.get("status") == "searched":
        out["status"], out["search_scope"] = NOT_FOUND, searched.get("scope")
    else:
        out["status"] = NOT_RUN
        out["note"] = "a detector exists for this class but no observation of it is available for this case"
    if searched:
        out["search_scope"] = searched.get("scope")
    return out


# ------------------------------------------------------------------ C. purpose fit + semantic verdict
def semantic_verdict(class_evs, claims, purpose, registry):
    """Returns (verdict, reasons, purpose_fit). The VLM's function description plays no part."""
    req = set(purpose["requires_any_of"])
    ev = [c for c in class_evs if c["status"] != NOT_EVIDENCE]
    established = [c for c in ev if c["status"] == VERIFIED_PRESENT]
    related = [c for c in established if c["n_related"] > 0]
    pr = [c for c in ev if set(c["capabilities"]) & req]                      # purpose-relevant classes the VLM cited
    pr_est = [c for c in pr if c["status"] == VERIFIED_PRESENT]
    n_unv = sum(1 for c in claims if c["kind"] != "registry")
    fit = {"purpose_known": purpose["known"], "requires_any_of": sorted(req), "established_capabilities": sorted(
        {cap for c in related for cap in c["capabilities"]}), "fit": PURPOSE_FIT_UNDETERMINED, "reasons": []}
    missing = [c for c in pr if c["status"] == NOT_FOUND]
    if missing:
        r = [f"cited '{c['class']}' (needed for the purpose) was searched for and not found" for c in missing]
        return EVIDENCE_NOT_ESTABLISHED, r, {**fit, "reasons": r}
    if not established:
        r = [f"no cited object could be verified: {sum(1 for c in ev if c['status'] == NOT_FOUND)} not found, "
             f"{sum(1 for c in ev if c['status'] == NOT_RUN)} not run, {n_unv} claim(s) UNVERIFIABLE (no detector)"]
        return EVIDENCE_NOT_ESTABLISHED, r, {**fit, "reasons": r}
    if purpose["known"] and pr_est:
        if any(c["n_related"] > 0 for c in pr_est):
            r = [f"verified '{c['class']}' stands in a relation to the region and carries "
                 f"{sorted(set(c['capabilities']) & req)}" for c in pr_est if c["n_related"] > 0]
            return SEMANTICALLY_SUPPORTED, r, {**fit, "fit": "SUPPORTED", "reasons": r}
        r = [f"verified '{c['class']}' exists but none of its {len(c['instances'])} instance(s) is inside or adjacent to the "
             "proposed region" for c in pr_est]
        return EVIDENCE_RELATION_INCONSISTENT, r, {**fit, "reasons": r}
    if not related:
        r = [f"verified '{c['class']}' exists but none of its instances is inside or adjacent to the region" for c in established]
        return EVIDENCE_RELATION_INCONSISTENT, r, {**fit, "reasons": r}
    if not purpose["known"]:
        r = ["the instruction names no purpose the table knows, so no capability can be required"]
    elif not (req & {cap for spec in registry.values() for cap in spec["capabilities"]}):
        r = [f"the purpose needs {sorted(req)}, which no registered detector class can verify"]
    else:
        r = [f"verified, related evidence supports {fit['established_capabilities']}, not the capability the purpose needs "
             f"({sorted(req)})"]
    return PURPOSE_FIT_UNDETERMINED, r, {**fit, "reasons": r}


# ------------------------------------------------------------------ D. placement readiness (never uses region size)
def region_conflicts(region_box, cited_classes, obs, pcfg, size, registry):
    problems, notes = [], []
    for o in obs["objects"]:
        if registry.get(o["class_name"], {}).get("role") == "equipment" and o["class_name"] not in cited_classes:
            c = containment(o["bbox_xyxy"], region_box)
            if c >= pcfg["max_other_equipment_containment"]:
                problems.append({"object": o["object_id"], "class": o["class_name"], "fraction_inside_region": round(c, 3),
                                 "reason": "the region contains equipment the VLM did not cite as evidence"})
    m = obs["map"]
    if m.get("alignment") == "ALIGNED" and m.get("layers"):
        x1, y1, x2, y2 = (int(round(v)) for v in region_box)
        for name, limit in (("traversable", pcfg["max_drive_zone_overlap_frac"]), ("no_go", pcfg["max_no_go_overlap_frac"])):
            sub = m["layers"][name][max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            frac = float(sub.mean()) if sub.size else 0.0
            (problems if frac > limit else notes).append({"map_layer": name, "fraction_of_region": round(frac, 4), "limit": limit,
                                                           "basis": "map prior (DRAFT_UNCONFIRMED), independent of the VLM"})
    else:
        notes.append({"map_check": "UNVERIFIABLE", "reason": f"map alignment is {m.get('alignment')}"})
    return {"problems": problems, "notes": notes}


def placement_readiness(cand, class_evs, claims, obs, pcfg, size, legacy, registry):
    box = cand["box_px"]
    cited = {c for cl in claims for c in cl["classes"]}
    ev_boxes = [i["bbox_xyxy"] for c in class_evs if c["status"] == VERIFIED_PRESENT for i in c["instances"]]
    good = []
    for loc in obs.get("localizations", []):
        if loc["candidate_id"] != cand["id"] or loc["source"] not in pcfg["accepted_localization_sources"]:
            continue
        inside = containment(loc["box_px"], box)
        anchored = any(containment(e, loc["box_px"]) >= 0.5 or box_gap(e, loc["box_px"]) == 0 for e in ev_boxes)
        if inside >= pcfg["min_localization_inside_proposal"] and anchored:
            good.append({**loc, "fraction_inside_proposal": round(inside, 3)})
    conflicts = region_conflicts(box, cited, obs, pcfg, size, registry)
    reasons = []
    if not good:
        reasons.append("no verified localization of a placeable spot inside the semantic region "
                       "(evidence-grounded refinement or a human-confirmed region)")
    if conflicts["problems"]:
        reasons.append("the region conflicts with other equipment or with the robot's travel/no-go zones")
    w, h = size
    return {"readiness": "ESTABLISHED" if (good and not conflicts["problems"]) else "NOT_ESTABLISHED", "reasons": reasons,
            "verified_localizations": good, "conflicts": conflicts,
            "region_size_info": {"area_px": round(box_area(box)), "area_frac_of_image": round(box_area(box) / (w * h), 4),
                                 "role": "reported only: NOT used for the semantic verdict or for readiness"},
            "legacy_v1_size_gate": legacy}


# ------------------------------------------------------------------ E. can the existing planning chain consume it
def planner_support(claims, class_evs):
    unmatched = [c["text"] for c in claims if c["kind"] == "unmatched_object"]
    not_run_obstacles = [c["class"] for c in class_evs if c["status"] == NOT_RUN and c["obstacle"]]
    incomplete = bool(unmatched or not_run_obstacles)
    carriers = [c for c in class_evs if c["status"] == VERIFIED_PRESENT and c["n_related"] and "truck" == c["class"]]
    return {
        "interface": "v0.2 placement_region / select_goal / A* take any region polygon (adapter route); the destination is "
                     "NOT converted into a truck-anchored loading_area",
        "native_v02_destination_stage": "truck-anchored loading_area only (agent/task_parser + semantics/region_grounding)",
        "native_type_supported": bool(carriers),
        "obstacle_model": "placement keeps out detected truck/conveyor/mobile_robot, verified objects the adapter passes in, "
                          "and map obstacles; an object it does not know is placeable floor to it",
        "obstacle_coverage": "INCOMPLETE" if incomplete else "COMPLETE",
        "unmodeled_physical_claims": unmatched, "classes_cited_but_not_observed": not_run_obstacles,
        "status": "UNSUPPORTED_OBSTACLE_COVERAGE_INCOMPLETE" if incomplete else "SUPPORTED",
    }


# ------------------------------------------------------------------ one candidate / one case
def verify_candidate(cand, obs, instruction, cfg, v02cfg, size):
    reg = cfg["registry"]
    purpose = purpose_from_instruction(instruction, cfg["purposes"])
    claims = extract_claims(cand["observed_evidence"], reg, cfg["marking_terms"])
    classes = sorted({c for cl in claims for c in cl["classes"]})
    class_evs = [class_evidence(c, cand["box_px"], obs, reg, v02cfg) for c in classes]
    for cl in claims:                                           # claim-level status for the report
        subs = [e for e in class_evs if e["class"] in cl["classes"]]
        cl["status"] = (UNVERIFIABLE if cl["kind"] != "registry" else
                        NOT_EVIDENCE if all(s["status"] == NOT_EVIDENCE for s in subs) else
                        VERIFIED_PRESENT if any(s["status"] == VERIFIED_PRESENT for s in subs) else
                        NOT_FOUND if any(s["status"] == NOT_FOUND for s in subs) else NOT_RUN)
        cl["relation"] = (HOLDS if any(s["n_related"] for s in subs) else FAILS) if cl["status"] == VERIFIED_PRESENT else None
    verdict, why, fit = semantic_verdict(class_evs, claims, purpose, reg)
    placement = placement_readiness(cand, class_evs, claims, obs, cfg["placement"], size, legacy_size_gate(cand["box_px"], obs, cfg), reg)
    planner = planner_support(claims, class_evs)
    blockers = []
    if verdict == SEMANTICALLY_SUPPORTED:
        if placement["readiness"] != "ESTABLISHED":
            blockers.append(PLACEMENT_UNVERIFIED)
        if planner["status"] != "SUPPORTED":
            blockers.append(PLANNER_UNSUPPORTED)
        final = blockers[0] if blockers else READY
    else:
        final = verdict
    pre = obs.get("preconditions", {})
    pre_block = []
    if obs["map"].get("alignment") != "ALIGNED":
        pre_block.append(f"map alignment: {obs['map'].get('alignment')}")
    if pre.get("cube") not in (None, "OK"):
        pre_block.append(f"red cube: {pre.get('cube')}")
    return {
        "candidate_id": cand["id"], "vlm_function_claim": cand.get("function_claimed"),
        "vlm_function_claim_note": "the VLM's own wording: recorded, never used as evidence or to decide",
        "purpose": purpose, "claims": claims, "reasoning_only_mentions": reasoning_only_classes(cand.get("reasoning_text", ""), claims, reg),
        "class_evidence": class_evs, "purpose_fit": fit,
        "semantic_verdict": verdict, "semantic_verdict_ko": LABELS_KO[verdict], "semantic_reasons": why,
        "placement": placement, "planner_support": planner, "all_blockers_after_semantics": blockers,
        "final_disposition": final, "final_disposition_ko": LABELS_KO[final],
        "planning_preconditions": {"blocked_by": pre_block, "start_pose": pre.get("start"),
                                   "note": "separate from the semantic verdict"},
        "hand_over_to_planner": final == READY and not pre_block,
    }


def verify_case(case, cfg, v02cfg):
    size = tuple(case["image_size"])
    res = [verify_candidate(c, case["observations"], case["instruction"], cfg, v02cfg, size) for c in case["candidates"]]
    ready = [r["candidate_id"] for r in res if r["final_disposition"] == READY]
    sem = [r["candidate_id"] for r in res if r["semantic_verdict"] == SEMANTICALLY_SUPPORTED]
    return {"case_id": case["id"], "instruction": case["instruction"], "candidates": res,
            "chosen_by_vlm": case.get("chosen"), "semantically_supported_candidates": sem, "ready_candidates": ready,
            "selection": "NOT_MADE_BY_THE_VERIFIER: it never replaces the VLM's choice; a planner-side choice rule (v0.2: "
                         "lowest goal score) is geometric, not a semantic criterion" if len(res) > 1 else "single candidate"}
