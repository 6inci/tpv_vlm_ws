"""Loads a SAVED integration run into the case contract that verifier2 consumes, and records what V1 decided on it.
Read-only: nothing in the source runs is modified. No model is called.

case = {id, image_size, instruction, candidates[{id, function_claimed, box_px, observed_evidence, reasoning_text, confidence}],
        chosen, observations{objects, searched, localizations, map, preconditions}, legacy{}, v1{...}}
"""
import json
from pathlib import Path

import cv2
import numpy as np

V02_CLASSES = ("truck", "conveyor", "mobile_robot")


def _json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def norm_to_px(b, w, h):
    return [b[0] / 1000 * w, b[1] / 1000 * h, b[2] / 1000 * w, b[3] / 1000 * h]


def load_run(run_dir):
    d = Path(run_dir).resolve()
    return {"dir": d, "run_id": d.name, "integration": _json(d / "integration_result.json"), "result": _json(d / "result.json"),
            "detections": _json(d / "analysis/detections.json"), "structured": _json(d / "vlm_proposal/structured.json"),
            "raw": (d / "vlm_proposal/raw_response.txt").read_text(encoding="utf-8")}


def _rasterize(spec, key, size):
    w, h = size
    m = np.zeros((h, w), np.uint8)
    for z in spec[key]:
        cv2.fillPoly(m, [np.round(np.asarray(z["polygon"], dtype=np.float64)).astype(np.int32)], 1)
    return m.astype(bool)


def _steps(run):
    return {s["name"]: s for s in run["result"]["steps"]}


def v1_stored(run):
    i = run["integration"]
    return {"run_id": run["run_id"], "integration_status": i["integration_status"], "overrides": i.get("config_overrides", []),
            "semantic_decision": i["semantic_decision"], "checks": i["verification"]["checks"],
            "refinement_chosen": ((i["verification"]["refinement"] or {}).get("chosen"))}


def build_case(case_cfg, base_dir):
    base = Path(base_dir)
    primary = load_run(base / case_cfg["runs"][0])
    supp = [load_run(base / r) for r in case_cfg.get("supplementary_runs", [])]
    for s in supp:                                              # supplementary data only if it is the SAME image and the SAME reply
        same_img = s["result"]["frame"].get("source_sha256") == primary["result"]["frame"].get("source_sha256")
        if not (same_img and s["raw"] == primary["raw"]):
            raise ValueError(f"supplementary run {s['run_id']} is not the same image + identical VLM reply as {primary['run_id']}")
    ij, res = primary["integration"], primary["result"]
    w, h = res["frame"]["width"], res["frame"]["height"]
    st = primary["structured"]
    ta = st["task_analysis"]
    area = next(a for a in st["functional_areas"] if a["id"] == ta["destination_area_id"])
    cand = {"id": area["id"], "function_claimed": area["function_name"], "box_px": norm_to_px(area["bbox_2d"], w, h),
            "observed_evidence": list(area["observed_evidence"]), "confidence": area["confidence"],
            "reasoning_text": f"{area['inference']} {ta['reason']}"}
    steps = _steps(primary)
    objects, searched = [], {}
    for o in primary["detections"]["objects"]:
        if o.get("accepted"):
            v = (o.get("verification") or {}).get("status")
            objects.append({"object_id": o["object_id"], "class_name": o["class_name"], "bbox_xyxy": o["bbox_xyxy"],
                            "basis": f"v0.2 {o['source']} (score {o.get('score')}) + Qwen crop verification '{v}' "
                                     f"[run {primary['run_id']}]"})
    if steps["detect_objects"]["status"] == "OK":
        searched.update({c: {"status": "searched", "scope": "full_image (v0.2 detect_objects)"} for c in V02_CLASSES})
    cube = ij["cube"]
    if steps.get("inspect_target", {}).get("status") in ("OK", "FAILED") and cube["status"] in ("OK", "TARGET_NOT_FOUND"):
        searched["red_cube"] = {"status": "searched", "scope": "v0.2 inspect_target"}
        if cube["bbox_xyxy"]:
            objects.append({"object_id": "red_cube_1", "class_name": "red_cube", "bbox_xyxy": cube["bbox_xyxy"],
                            "basis": f"v0.2 target inspection [run {primary['run_id']}]"})
    for run in [primary] + supp:                                # ad-hoc detections that a run happened to reach
        for cls, rec in ((run["integration"]["verification"].get("cones") or {}).items()):
            if cls in searched:
                continue
            searched[cls] = {"status": "searched", "scope": "near the VLM proposal only (DINO candidates within a 15% margin)"}
            for k, b in enumerate(rec["verified"], 1):
                objects.append({"object_id": f"{cls}_{k}", "class_name": cls, "bbox_xyxy": b,
                                "basis": f"Grounding DINO + Qwen crop verification [run {run['run_id']}]"})
    chosen = ((ij["verification"]["refinement"] or {}).get("chosen"))
    locs = [] if not chosen else [{"candidate_id": cand["id"], "box_px": chosen["bbox_xyxy"], "query": chosen["query"],
                                   "source": "evidence_grounded_refinement"}]
    align = ((res.get("map") or {}).get("alignment") or {}).get("status", "NOT_RUN")
    layers = None
    if align == "ALIGNED":
        spec = res["map"]["spec"]
        layers = {"traversable": _rasterize(spec, "traversable", (w, h)), "no_go": _rasterize(spec, "no_go", (w, h))}
    return {
        "id": case_cfg["id"], "image_size": [w, h], "instruction": ij["input"]["instruction"], "candidates": [cand],
        "chosen": cand["id"], "other_areas_proposed_by_vlm": [
            {"id": a["id"], "function_name": a["function_name"]} for a in st["functional_areas"] if a["id"] != cand["id"]],
        "observations": {"objects": objects, "searched": searched, "localizations": locs,
                         "map": {"alignment": align, "layers": layers},
                         "preconditions": {"cube": cube["status"], "start": (ij["robot_start"] or {}).get("knowledge")}},
        "image_path": str(Path(res["frame"]["source"]["path"])),
        "vlm_raw_response_file": str(primary["dir"] / "vlm_proposal/raw_response.txt"),
        "v1": {"primary": v1_stored(primary), "supplementary": [v1_stored(s) for s in supp]},
        "runs": [primary["run_id"]] + [s["run_id"] for s in supp],
        "_v1_runs": [primary] + supp,
    }
