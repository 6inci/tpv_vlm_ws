"""Replays the V1 gate (experiments/semantic_to_navigation_integration) on stored or synthetic inputs, OFFLINE.

V1's gate lives inside IntegrationRun._infer_destination, interleaved with model calls, so it cannot be called without a
run. This module repeats the same sequence with the same pure functions from V1's verify.py (imported by path,
unchanged). Faithfulness is not assumed: for every STORED run the replay must reproduce the statuses V1 saved
(`reproduces_stored`), otherwise the comparison is reported as unreliable.
"""
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V1_DIR = (HERE.parent / "semantic_to_navigation_integration").resolve()


def load_v1():
    spec = importlib.util.spec_from_file_location("v1_verify", V1_DIR / "verify.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["v1_verify"] = mod
    spec.loader.exec_module(mod)
    return mod


def v1_config():
    import yaml
    return yaml.safe_load(open(V1_DIR / "config.yaml", encoding="utf-8"))


def icfg_with_overrides(icfg, overrides):
    """A stored run may have used `--set a.b.c=value` (a VARIANT). Replaying it needs the same value."""
    import copy
    import yaml
    icfg = copy.deepcopy(icfg)
    for item in overrides or []:
        key, _, val = item.partition("=")
        node = icfg
        parts = key.split(".")
        for k in parts[:-1]:
            node = node[k]
        node[parts[-1]] = yaml.safe_load(val)
    return icfg


def replay(v1, inp, icfg, v02cfg):
    """inp: area (VLM dict: function_name, inference, observed_evidence, id), vbox, size, accepted (v0.2 detections),
    refinement_chosen (box or None), cone_hits ({class: [boxes]}), judge ('yes'/'no'/None), map_aligned, layers, other_areas."""
    iv, dcfg, rcfg = icfg["verification"], v02cfg["destination"], v02cfg["region_grounding"]
    area, vbox, size = inp["area"], inp["vbox"], inp["size"]
    accepted = inp["accepted"]
    trucks = [o for o in accepted if o["class_name"] == "truck"]
    cited, unv = v1.cited_items(area, iv["evidence_vocabulary"])
    checks = []
    c1 = v1.image_link(vbox, size, iv)
    checks.append(c1)
    ok, truck_box, adopted, basis = c1["status"] == v1.VERIFIED, None, None, None
    if ok and len(trucks) != 1:
        ok = False
        checks.append(v1.check("V3", "region is geometrically associated with the verified anchor", v1.FAILED, "python_geometry",
                               {"reject_reason": f"{len(trucks)} trucks accepted by the v0.2 detector (exactly one is required as the anchor)"}))
    elif ok:
        truck_box = trucks[0]["bbox_xyxy"]
        v3 = v1.association(vbox, truck_box, dcfg, rcfg)
        checks.append(v3)
        ok = v3["status"] == v1.VERIFIED
    if ok:
        chosen = inp.get("refinement_chosen")
        if chosen:
            adopted, basis = chosen, "refined_by_evidence_grounding"
            st = v1.VERIFIED
        else:
            clipped = v1.inter_box(vbox, v1.association_zone(truck_box, size, rcfg))
            ratio = None if clipped is None else round(v1.box_area(clipped) / max(1.0, v1.box_area(truck_box)), 3)
            basis = "coarse_clipped_to_association_zone"
            if clipped is None or v1.box_area(clipped) < rcfg["min_candidate_area_px"]:
                st = v1.FAILED
            elif ratio > iv["max_unrefined_area_ratio"]:
                st = v1.FAILED
            else:
                st = v1.ESTIMATED
            adopted = clipped
        checks.append(v1.check("V3b", "spatial precision of the adopted region", st, "replay", {"basis": basis}))
        ok = st in (v1.VERIFIED, v1.ESTIMATED)
    cone_hits = inp.get("cone_hits", {}) if ok else {}
    v2 = v1.evidence_existence(cited, unv, accepted, cone_hits)
    checks.append(v2)
    ok = ok and v2["status"] == v1.VERIFIED
    if ok:
        others = [o for o in accepted if o["class_name"] != "truck"]
        v5 = v1.inclusion(adopted, others, inp.get("other_areas", []), inp.get("layers"), iv, size)
        checks.append(v5)
        ok = v5["status"] != v1.FAILED
    if ok and iv["judge"]["enabled"]:
        j = inp.get("judge")
        checks.append(v1.check("V4", "v0.2 region_judge accepts the adopted region", v1.ESTIMATED if j == "yes" else v1.FAILED,
                               "replay (judge answer taken from the stored run / scenario)", {"answer": j}))
        ok = j == "yes"
    elif iv["judge"]["enabled"]:
        checks.append(v1.check("V4", "v0.2 region_judge accepts the adopted region", v1.NOT_RUN, "replay", {}))
    status, reasons = v1.decide(checks, bool(inp.get("map_aligned")), basis)
    return {"checks": sorted(checks, key=lambda c: c["id"]), "decision": status, "reasons": reasons,
            "adopted_basis": basis}


def inputs_from_run(run, case):
    """Replay inputs for one stored run (the run's own refinement / cones / judge / map record)."""
    i = run["integration"]
    st = run["structured"]
    ta = st["task_analysis"]
    area = next(a for a in st["functional_areas"] if a["id"] == ta["destination_area_id"])
    w, h = case["image_size"]
    vbox = [area["bbox_2d"][0] / 1000 * w, area["bbox_2d"][1] / 1000 * h, area["bbox_2d"][2] / 1000 * w, area["bbox_2d"][3] / 1000 * h]
    accepted = [o for o in run["detections"]["objects"] if o.get("accepted")]
    chosen = (i["verification"]["refinement"] or {}).get("chosen")
    cones = {k: v["verified"] for k, v in (i["verification"].get("cones") or {}).items()}
    v4 = next((c for c in i["verification"]["checks"] if c["id"] == "V4"), None)
    judge = (v4["detail"] or {}).get("answer") if v4 and v4["status"] != "NOT_RUN" else None
    aligned = ((run["result"].get("map") or {}).get("alignment") or {}).get("status") == "ALIGNED"
    layers = case["observations"]["map"]["layers"] if aligned else None
    others = [{"id": a["id"], "function_name": a["function_name"],
               "box_px": [a["bbox_2d"][0] / 1000 * w, a["bbox_2d"][1] / 1000 * h, a["bbox_2d"][2] / 1000 * w, a["bbox_2d"][3] / 1000 * h]}
              for a in st["functional_areas"] if a["id"] != area["id"]]
    return {"area": area, "vbox": vbox, "size": (w, h), "accepted": accepted,
            "refinement_chosen": chosen["bbox_xyxy"] if chosen else None, "cone_hits": cones, "judge": judge,
            "map_aligned": aligned, "layers": layers, "other_areas": others}


def reproduces(stored, replayed):
    """Do the statuses V1 stored equal the replay's, check by check (V4 is taken from the stored record)?"""
    s = {c["id"]: c["status"] for c in stored["checks"]}
    r = {c["id"]: c["status"] for c in replayed["checks"]}
    same_checks = s == r
    stored_dec = stored["semantic_decision"]["status"]
    mapping = {"WITHHELD": "SEMANTIC_NOT_VERIFIED"}
    same_dec = mapping.get(replayed["decision"], replayed["decision"]) == stored_dec
    return {"checks_equal": same_checks, "decision_equal": same_dec, "stored": s, "replayed": r,
            "stored_decision": stored_dec, "replayed_decision": replayed["decision"]}
