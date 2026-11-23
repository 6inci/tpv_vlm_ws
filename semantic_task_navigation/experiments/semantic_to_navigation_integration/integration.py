#!/usr/bin/env python3
"""Semantic-to-navigation integration prototype (v0.2 is subclassed, never edited).

  STEP 1  instruction (L2)                       config.yaml, equals the frozen natural_language_abstraction L2 text
  STEP 2  external camera image                  v0.2 ImageSource / input/raw_frame.png
  STEP 3  Qwen3-VL free proposal of a destination   nla_probe.run_level (frozen prompt n0): image + instruction ONLY
  STEP 4  verification of proposal + evidence    verify.py + v0.2 detections, ad-hoc DINO+Qwen crop check, v0.2 region_judge
  STEP 5  adopt or withhold                      verify.decide (required checks must pass)
  STEP 6  placement region                       v0.2 semantics.placement.placement_region
  STEP 7  robot approach goal                    v0.2 navigation.goal_selector.select_goal (reach is PROVISIONAL)
  STEP 8  A* path + independent validation       v0.2 navigation.planner / path_validator
  STEP 9  overlay                                visualize.py

Overridden AnalysisRun methods (everything else is v0.2 code as it is): _parse_task (v0.2's parser can only express "loading
area anchored on a truck", which would inject the answer), _analyze_scene (its output is never used by v0.2 and its prompt
would carry a v0.2 task summary), _infer_destination (steps 3-5), _assess_placement (adds verified obstacles),
_missing (planning needs an ALIGNED map). The VLM never sees detections, the map, zones or any answer region.

  PY=/home/binci/miniconda3/envs/stnav_gui/bin/python
  PYTHONDONTWRITEBYTECODE=1 $PY experiments/semantic_to_navigation_integration/integration.py run --image-id DEV_REF
"""
import argparse
import copy
import importlib.util
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import yaml

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(HERE))

from semantic_task_nav.agent import task_parser  # noqa: E402
from semantic_task_nav.agent.orchestrator import AnalysisRun, StepFailure  # noqa: E402
from semantic_task_nav.config import (code_fingerprint, load_config, resolve_path, sha256_file,  # noqa: E402
                                      sha256_text)
from semantic_task_nav.contracts import MANUAL, OBSERVED, RAW_PX, REGION_SCHEMA, TASK_SCHEMA, validate  # noqa: E402
from semantic_task_nav.perception import grounding  # noqa: E402
from semantic_task_nav.perception.camera import ImageSource  # noqa: E402
from semantic_task_nav.perception.scene_graph import add_region_relation  # noqa: E402
from semantic_task_nav.semantics import region_grounding as rg  # noqa: E402
from semantic_task_nav.semantics.placement import placement_region  # noqa: E402

import verify  # noqa: E402
from verify import ESTIMATED, FAILED, NOT_RUN, VERIFIED  # noqa: E402


def load_icfg():
    with open(HERE / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_overrides(icfg, sets):
    """--set a.b.c=value (YAML value). A VARIANT run: recorded in the record, the manifest and the overlay; config.yaml is not edited."""
    icfg = copy.deepcopy(icfg)
    for item in sets or []:
        key, _, val = item.partition("=")
        node = icfg
        parts = key.split(".")
        for k in parts[:-1]:
            node = node[k]
        if parts[-1] not in node:
            raise SystemExit(f"--set: unknown key {key}")
        node[parts[-1]] = yaml.safe_load(val)
    icfg["_overrides"] = list(sets or [])
    return icfg


def load_nla(icfg):
    path = (HERE / icfg["vlm"]["experiment_dir"]).resolve() / "nla_probe.py"
    spec = importlib.util.spec_from_file_location("nla_probe", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["nla_probe"] = mod
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ model-using tools (replaceable in tests)
class Tools:
    """Everything that calls a model. IntegrationRun only talks to these four methods."""

    def __init__(self, client, cfg, icfg):
        self.client, self.cfg, self.icfg = client, cfg, icfg
        self.nla = load_nla(icfg)
        self.nla_exp, self.nla_prompts = self.nla.load_experiment()
        ok, msg = self.nla.check_frozen(self.nla_exp, self.nla_prompts)
        self.frozen = {"ok": ok, "message": msg, "level": icfg["instruction_level"]}
        if icfg["vlm"]["require_frozen_prompt"] and not ok:
            raise SystemExit(f"refusing to run: {msg}")
        if self.nla_exp["instruction_levels"][icfg["instruction_level"]] != icfg["instruction"]:
            raise SystemExit("instruction differs from the frozen natural_language_abstraction level text")

    def propose(self, writer, frame_info):
        """STEP 3. One image + the frozen prompt; nothing else reaches the model."""
        img_rec = {"id": "integration", "role": "integration", "file": str(writer.path("input/raw_frame.png")),
                   "sha256": frame_info.get("source_sha256") or frame_info["decoded_pixels_sha256"]}
        res = self.nla.run_level(self.client, self.nla_exp, self.nla_prompts, img_rec, self.icfg["instruction_level"],
                                 writer.path("vlm_proposal"))
        return res

    def ground(self, image_path, size, query):
        return grounding.qwen_ground_boxes(self.client, self.cfg["models"], image_path, size, query)

    def detect(self, spec, image, image_path, size, region_box, margin_frac, crop_dir):
        """Ad-hoc object class near the proposal: Grounding DINO candidates, each confirmed by the v0.2 Qwen crop check."""
        mcfg = self.cfg["models"]
        cands, _ = grounding.dino_candidates(self.client, mcfg, image_path, size, spec["dino_prompts"])
        mx, my = margin_frac * (region_box[2] - region_box[0]), margin_frac * (region_box[3] - region_box[1])
        ex = [region_box[0] - mx, region_box[1] - my, region_box[2] + mx, region_box[3] + my]
        near = [c for c in cands if ex[0] <= (c["bbox_xyxy"][0] + c["bbox_xyxy"][2]) / 2 <= ex[2]
                and ex[1] <= (c["bbox_xyxy"][1] + c["bbox_xyxy"][3]) / 2 <= ex[3]]
        near = sorted(near, key=lambda c: -(c["score"] or 0))[:spec["top_k"]]
        recs, verified = [], []
        for i, c in enumerate(near, 1):
            v = grounding.verify_box(self.client, mcfg, image, c["bbox_xyxy"], spec["description"],
                                     Path(crop_dir) / f"crop_{i}.png")
            recs.append({"bbox_xyxy": [round(x, 1) for x in c["bbox_xyxy"]], "score": c["score"], "status": v["status"]})
            if v["status"] == "yes":
                verified.append([round(x, 1) for x in c["bbox_xyxy"]])
        return {"n_dino_candidates": len(cands), "n_near_proposal": len(near), "checked": recs, "verified": verified}

    def judge(self, image, truck_box, cands, out_dir, margin):
        rg.judge_candidates(self.client, self.cfg["models"], image, truck_box, cands, Path(out_dir), margin)


# ------------------------------------------------------------------ the run
class IntegrationRun(AnalysisRun):
    PLANNING_STEPS = ("assess_placement", "select_pickup_pose", "select_dropoff_pose", "plan_path", "validate_path",
                      "create_handoff_request")

    def __init__(self, cfg, icfg, mode, frame, command, client, tools, **kw):
        super().__init__(cfg, mode, frame, command, client, **kw)
        self.icfg, self.tools = icfg, tools
        if hasattr(tools, "client"):          # route the semantic-step model calls through v0.2's TrackedClient, so every
            tools.client = self.client        # call is logged in analysis/model_calls.jsonl like the v0.2 steps
        self.integ = {"vlm": None, "checks": [], "refinement": None, "cones": None, "semantic": None, "adopted": None}

    # -------------------------------------------------------- replaced v0.2 steps
    def _parse_task(self):
        """v0.2's parser accepts only 'loading_area anchored on a truck' (task_parser.build_task). Using it would put the
        answer into the task. Only the TARGET is parsed, with v0.2's own keyword rules; the destination stays open."""
        obj = task_parser.rule_based_parse(self.command)
        t = obj.get("target")
        record = {"command": self.command, "mode": self.mode, "status": "OK", "backend": "integration_adapter",
                  "v02_rule_based_parse": obj}
        if obj.get("task_type") != "transport_and_place" or not t or t["count"] != 1:
            record.update(status="FAILED", error_code="UNSUPPORTED_TASK", error="no single transportable target found")
            self.writer.write_json("analysis/task_parse.json", record)
            self.ctx["task_parse_error"] = record
            raise StepFailure("UNSUPPORTED_TASK", record["error"])
        cls = task_parser.CLASS_SYNONYMS.get(t["class"].strip().lower())
        color = (t["color"] or "").strip().lower()
        if cls is None or color not in task_parser.SUPPORTED_TARGET_COLORS:
            record.update(status="FAILED", error_code="UNSUPPORTED_TARGET", error=f"{t['color']} {t['class']}")
            self.writer.write_json("analysis/task_parse.json", record)
            self.ctx["task_parse_error"] = record
            raise StepFailure("UNSUPPORTED_TARGET", record["error"])
        task = validate({
            "task_type": "transport_and_place",
            "target": {"class": cls, "color": color, "source_reference": None, "source_relation": None},
            "destination": {"region_type": "UNSPECIFIED (inferred by the VLM from image + instruction)",
                            "reference_object": None},
            "actions": list(task_parser.ACTION_TEMPLATES["transport_and_place"]),
            "actions_source": "fixed_template (not model output)",
            "parser": {"backend": "integration_adapter(v0.2 rule_based_parse for the target only)", "model": None,
                       "prompt_id": None, "raw_response": json.dumps(obj, ensure_ascii=False), "parsed_output": obj,
                       "normalization_notes": ["destination deliberately not parsed: the instruction text goes to the VLM"]},
        }, TASK_SCHEMA)
        record.update(task=task, call=None)
        self.writer.write_json("analysis/task_parse.json", record)
        self.ctx["task"] = task
        return {"model": "v0.2 keyword rules (target only)", "input": self.command,
                "output": f"transport_and_place: target=red cube; destination=UNSPECIFIED (left to the VLM)"}

    def _analyze_scene(self):
        self.ctx["scene_analysis"] = {"status": "NOT_RUN", "reason": "v0.2 never uses this output; its prompt would insert a "
                                      "v0.2 task summary naming a loading area near a truck"}
        return {"status": "NOT_RUN", "model": "none", "output": "not used by the integration"}

    def _missing(self, name):
        m = super()._missing(name)
        if name in self.PLANNING_STEPS and "nav" not in self.ctx:
            m = ["nav (map not aligned with this image, or not built: planning stopped)"] + [x for x in m if x != "nav"]
        return m

    def _assess_placement(self):
        """v0.2 placement_region, plus verified obstacle objects (cones) that v0.2 has no detector class for."""
        h, w = self.frame.image.shape[:2]
        acc = [o for o in self.ctx["detections"]["objects"] if o.get("accepted")] + list(self.ctx.get("extra_objects", []))
        pcfg = copy.deepcopy(self.cfg["planning"])
        pcfg["placement"]["exclude_detected_classes"] = list(pcfg["placement"]["exclude_detected_classes"]) + sorted(
            {o["class_name"] for o in self.ctx.get("extra_objects", [])})
        recs = {r["region_id"]: placement_region(r, (w, h), acc, self.ctx.get("target"), self.ctx.get("nav"), pcfg)
                for r in self.ctx["semantic_regions"]}
        self.ctx["placements"] = recs
        self.writer.write_json("analysis/placement_regions.json", recs)
        summary = "; ".join(f"{k}: {v['status']} ({v['area_px']} px)" for k, v in recs.items())
        if not any(v["status"] == "PLACEMENT_REGION_FOUND" for v in recs.values()):
            raise StepFailure("PLACEMENT_REGION_EMPTY", summary)
        return {"model": "geometry (adopted region - object footprints - verified cones - map obstacles, eroded)",
                "input": f"{len(recs)} adopted region(s)", "output": summary + " | surface & reach UNMEASURED"}

    # -------------------------------------------------------- STEP 3-5
    def _stop(self, semantic_status, reasons, code, step_status="UNCERTAIN", **extra):
        self.integ["semantic"] = {"status": semantic_status, "reasons": reasons, **extra}
        self.ctx["region_grounding"] = {"status": semantic_status, "reason": "; ".join(reasons), "loading_area_candidates": [],
                                        "semantic_regions": [], "selected_loading_region": None,
                                        "validated_loading_region": None, "prior_zone_consistency": [],
                                        "note": "integration: VLM free proposal + verification; nothing passed to placement"}
        self._write_verification()
        raise StepFailure(code, "; ".join(reasons), status=step_status)

    def _write_verification(self):
        self.writer.write_json("verification/verification.json", {
            "note": "Python / detector checks of the VLM proposal. The VLM output itself is in vlm_proposal/.",
            "checks": sorted(self.integ["checks"], key=lambda c: c["id"]), "refinement": self.integ["refinement"],
            "cones": self.integ["cones"], "semantic_decision": self.integ["semantic"], "adopted": self.integ["adopted"]})

    def _infer_destination(self):
        iv, pcfg = self.icfg["verification"], self.cfg["planning"]
        dcfg, rcfg = pcfg["destination"], pcfg["region_grounding"]
        img = self.frame.image
        h, w = img.shape[:2]
        size = (w, h)
        raw_png = self.writer.path("input/raw_frame.png")
        I, checks = self.integ, self.integ["checks"]

        # ---- STEP 3: the VLM proposes (image + instruction only)
        prop = self.tools.propose(self.writer, self.ctx["frame_info"])
        I["vlm"] = {"status": prop["status"], "error": prop.get("error"), "decision": prop.get("decision"),
                    "prompt_sha256": prop["prompt_sha256"], "model_input_audit": prop["model_input_audit"],
                    "call": prop["call"], "consistency_issues": prop["consistency_issues"],
                    "files": "vlm_proposal/{prompt.txt,raw_response.txt,structured.json,result.json,overlay_areas.png}"}
        if prop["status"] != "OK":
            self._stop("VLM_RESPONSE_INVALID", [str(prop["error"])], "VLM_RESPONSE_INVALID", "FAILED")
        s = prop["structured"]
        area, vbox = verify.destination_of(s, size)
        I["vlm"]["destination"] = None if area is None else {
            "id": area["id"], "function_name": area["function_name"], "confidence": area["confidence"],
            "box_px": [round(v, 1) for v in vbox], "observed_evidence": area["observed_evidence"],
            "inference": area["inference"], "reason": s["task_analysis"]["reason"],
            "info_given_by_instruction": s["task_analysis"]["info_given_by_instruction"],
            "info_added_by_you": s["task_analysis"]["info_added_by_you"]}
        if area is None:
            ta = s["task_analysis"]
            why = ta["reason"] if ta["decision"] != "proposed" else "decision=proposed but destination_area_id is not an area"
            self._stop("VLM_WITHHELD" if ta["decision"] != "proposed" else "VLM_PROPOSAL_INCONSISTENT", [why],
                       "SEMANTIC_NOT_INFERRED")

        # ---- STEP 4: verification (the VLM is not asked to confirm itself, except the labelled v0.2 judge)
        vocab = iv["evidence_vocabulary"]
        accepted = [o for o in self.ctx["detections"]["objects"] if o.get("accepted")]
        trucks = [o for o in accepted if o["class_name"] == "truck"]
        cited, unverifiable = verify.cited_items(area, vocab)
        c1 = verify.image_link(vbox, size, iv)
        checks.append(c1)
        ok, truck_box, adopted_box, basis = c1["status"] == VERIFIED, None, None, None
        if ok and len(trucks) != 1:
            ok = False
            checks.append(verify.check("V3", "region is geometrically associated with the verified anchor", FAILED,
                                       "python_geometry", {"reject_reason": f"{len(trucks)} trucks accepted by the v0.2 detector "
                                                           "(exactly one is required as the anchor)"}))
        elif ok:
            truck_box = trucks[0]["bbox_xyxy"]
            v3 = verify.association(vbox, truck_box, dcfg, rcfg)
            checks.append(v3)
            ok = v3["status"] == VERIFIED
        if ok:                                                    # tighten using the VLM's OWN cited evidence
            I["refinement"] = self._refine(area, vbox, truck_box, cited, size, raw_png)
            chosen = I["refinement"]["chosen"]
            if chosen:
                adopted_box, basis = chosen["bbox_xyxy"], "refined_by_evidence_grounding"
                v3b = verify.check("V3b", "spatial precision of the adopted region", VERIFIED,
                                   "vlm grounding of the VLM's own evidence + python geometry (v0.2 association rules)",
                                   {"basis": basis, "query": chosen["query"], "area_ratio_to_truck": chosen["area_ratio_to_truck"]})
            else:
                clipped = verify.inter_box(vbox, verify.association_zone(truck_box, size, rcfg))
                ratio = None if clipped is None else round(verify.box_area(clipped) / max(1.0, verify.box_area(truck_box)), 3)
                basis = "coarse_clipped_to_association_zone"
                if clipped is None or verify.box_area(clipped) < rcfg["min_candidate_area_px"]:
                    st, why = FAILED, "nothing of the proposal lies within the truck association zone"
                elif ratio > iv["max_unrefined_area_ratio"]:
                    st, why = FAILED, (f"proposal is {ratio}x the truck box and could not be tightened by evidence grounding "
                                       f"(limit {iv['max_unrefined_area_ratio']}x without refinement)")
                else:
                    st, why = ESTIMATED, "tight enough to use without refinement; its edges are not verified"
                adopted_box = clipped
                v3b = verify.check("V3b", "spatial precision of the adopted region", st,
                                   "python_geometry (box clipped to the truck association zone)",
                                   {"basis": basis, "area_ratio_to_truck": ratio, "note": why,
                                    "clipped_box_px": None if clipped is None else [round(v, 1) for v in clipped],
                                    "refinement_attempts": len(I["refinement"]["attempts"])})
            checks.append(v3b)
            ok = v3b["status"] in (VERIFIED, ESTIMATED)
        cone_hits = {}
        if ok:                                                    # objects that must stay out of the placement region
            I["cones"] = {}
            for cname in iv["always_detect"]:
                res = self.tools.detect(vocab[cname], img, raw_png, size, adopted_box, iv["detect_region_margin_frac"],
                                        self.writer.path(f"verification/{cname}_crops"))
                I["cones"][cname] = res
                cone_hits[cname] = res["verified"]
        v2 = verify.evidence_existence(cited, unverifiable, accepted, cone_hits)
        checks.append(v2)
        ok = ok and v2["status"] == VERIFIED
        if ok:
            others = [o for o in accepted if o["class_name"] != "truck"]
            areas = [{"id": a["id"], "function_name": a["function_name"], "box_px": verify.norm_to_px(a["bbox_2d"], w, h)}
                     for a in s["functional_areas"] if a["id"] != area["id"]]
            nav = self.ctx.get("nav")
            v5 = verify.inclusion(adopted_box, others, areas, nav.full_layers if nav is not None else None, iv, size)
            checks.append(v5)
            ok = v5["status"] != FAILED
        cands = []
        if ok and iv["judge"]["enabled"]:
            cands = [{"candidate_id": 1, "polygon_xy": verify.rect(adopted_box), "side_of_truck_box": "vlm_proposal",
                      "source": f"vlm_free_proposal:{basis}", "coordinate_frame": RAW_PX,
                      "generation": "adopted region (VLM proposal after verification)"}]
            if iv["judge"]["controls"]:
                for c in rg.generate_candidates(truck_box, size, rcfg):
                    c = dict(c, candidate_id=len(cands) + 1, control=True)
                    cands.append(c)
            self.tools.judge(img, truck_box, cands, self.writer.path("verification/judge"), dcfg["judge_crop_margin_px"])
            ans = cands[0]["vlm_plausible"]
            yes_controls = [c["candidate_id"] for c in cands[1:] if c.get("vlm_plausible") == "yes"]
            checks.append(verify.check(
                "V4", "v0.2 region_judge accepts the adopted region", ESTIMATED if ans == "yes" else FAILED,
                "vlm_judge_same_model (v0.2 region_judge_v1; its loading-area definition is human-written; not independent)",
                {"answer": ans, "reason": cands[0]["vlm_reason"],
                 "controls": {c["candidate_id"]: {"side": c.get("side_of_truck_box"), "answer": c.get("vlm_plausible")}
                              for c in cands[1:]},
                 "judge_discriminates": not yes_controls,
                 "note": "yes on a region that contains the truck carries little information; controls are recorded, not gating"}))
            ok = ans == "yes"
        elif iv["judge"]["enabled"]:
            checks.append(verify.check("V4", "v0.2 region_judge accepts the adopted region", NOT_RUN, "vlm_judge_same_model",
                                       {"reason": "an earlier required check failed"}))

        map_aligned = self.ctx.get("nav") is not None
        # ---- STEP 5: adopt or withhold
        status, reasons = verify.decide(checks, map_aligned, basis)
        I["adopted"] = None if adopted_box is None else {
            "basis": basis, "box_px": [round(v, 1) for v in adopted_box], "polygon_xy": verify.rect(adopted_box),
            "vlm_box_px": [round(v, 1) for v in vbox], "truck_box_px": truck_box and [round(v, 1) for v in truck_box],
            "fraction_of_vlm_box_kept": round(verify.box_area(adopted_box) / max(1.0, verify.box_area(vbox)), 3)}
        if status == "WITHHELD":
            self._stop("SEMANTIC_NOT_VERIFIED", reasons, "SEMANTIC_NOT_VERIFIED",
                       evidence_failed=any(c["id"] == "V2" and c["status"] != VERIFIED for c in checks))
        truck = trucks[0]
        region = validate({
            "region_id": "loading_area_candidate_1", "region_type": "vlm_inferred_destination",
            "polygon_xy": verify.rect(adopted_box), "coordinate_frame": RAW_PX, "reference_objects": [truck["object_id"]],
            "evidence": list(area["observed_evidence"]) + [f"python verification: {c['id']} {c['status']}" for c in checks],
            "verification_status": "unverified",   # the semantic meaning stays a VLM inference; checks cover geometry/evidence only
            "source": f"qwen3_vl_free_proposal+python_verification:{basis}"}, REGION_SCHEMA)
        I["semantic"] = {"status": status, "reasons": reasons, "region_id": region["region_id"]}
        self.ctx["extra_objects"] = [
            {"object_id": f"traffic_cone_{i}", "class_name": "traffic_cone", "bbox_xyxy": b, "accepted": True,
             "knowledge": OBSERVED, "coordinate_frame": RAW_PX, "source": "grounding_dino+qwen_crop_verification"}
            for i, b in enumerate(cone_hits.get("traffic_cone", []), 1)]
        self._write_verification()
        if status.startswith("ADOPTED_SEMANTIC_ONLY"):
            self.ctx["region_grounding"] = {"status": "SEMANTIC_ADOPTED_PLANNING_STOPPED_MAP_NOT_ALIGNED",
                                            "loading_area_candidates": [], "semantic_regions": [],
                                            "selected_loading_region": None, "validated_loading_region": None,
                                            "prior_zone_consistency": [], "adopted_region_not_passed_on": region,
                                            "note": "adopted for semantics only; the map is not aligned with this image"}
            return {"model": "Qwen3-VL proposal + python verification", "input": "image + instruction",
                    "output": f"{status}: {region['region_id']} basis={basis}; planning stopped (map not aligned)"}
        res = self._rel({"method": "integration: Qwen3-VL free proposal -> python verification (no candidate list)",
                         "status": "SEMANTIC_REGION_FOUND", "truck_object": truck["object_id"], "loading_area_candidates": cands,
                         "structure_grounding": [], "semantic_regions": [region], "selected_loading_region": region,
                         "validated_loading_region": None, "prior_zone_consistency": [],
                         "note": "candidate 1 = adopted VLM proposal; candidates >1 (if any) are judge negative controls only"})
        self.ctx["region_grounding"] = res
        self.ctx["semantic_regions"] = [region]
        if "scene_graph" in self.ctx:
            add_region_relation(self.ctx["scene_graph"], truck["object_id"], region, "INFERRED")
        return {"model": "Qwen3-VL proposal + python verification", "input": "image + instruction",
                "output": f"{status}: {region['region_id']} basis={basis}; checks=" +
                          ",".join(f"{c['id']}:{c['status']}" for c in sorted(checks, key=lambda c: c['id']))}

    def _refine(self, area, vbox, truck_box, cited, size, raw_png):
        """Ground the VLM's own phrases (never a human-written query) and keep a box only if it passes the v0.2 association
        rules, lies inside the VLM box and encloses the verified truck. Several accepted -> the tightest."""
        iv, pcfg = self.icfg["verification"], self.cfg["planning"]
        rf = iv["refine"]
        out = {"enabled": rf["enabled"], "queries": [], "attempts": [], "chosen": None}
        if not rf["enabled"]:
            return out
        _, unverifiable = verify.cited_items(area, iv["evidence_vocabulary"])
        queries = []
        for q in unverifiable + [area["function_name"]]:
            q = q.strip()[:rf["max_query_chars"]]
            if q and q not in queries:
                queries.append(q)
        out["queries"] = queries = queries[:rf["max_queries"]]
        for q in queries:
            boxes, rec = self.tools.ground(raw_png, size, q)
            att = {"query": q, "raw_response": rec["raw_response"], "error": rec["error"], "boxes": []}
            for b in boxes:
                bb = b["bbox_xyxy"]
                okb, why = verify.screen_refined(bb, vbox, truck_box, pcfg["destination"], pcfg["region_grounding"], rf)
                att["boxes"].append({"bbox_xyxy": [round(v, 1) for v in bb], "accepted": okb, "reject_reason": why})
                if okb and verify.box_area(bb) >= pcfg["region_grounding"]["min_candidate_area_px"]:
                    out["attempts"].append({"query": q, "bbox_xyxy": bb})
            out.setdefault("log", []).append(att)
        if out["attempts"]:
            best = min(out["attempts"], key=lambda a: verify.box_area(a["bbox_xyxy"]))
            out["chosen"] = {"query": best["query"], "bbox_xyxy": best["bbox_xyxy"],
                             "area_ratio_to_truck": round(verify.box_area(best["bbox_xyxy"]) / max(1.0, verify.box_area(truck_box)), 3)}
        return out


# ------------------------------------------------------------------ integration record
def _step(result, name):
    return next((s for s in result["steps"] if s["name"] == name), None)


def build_record(run, image_spec, tools, v02_before, v02_after):
    r, I, ctx = run.result, run.integ, run.ctx
    amap = (ctx.get("map_record") or {}).get("alignment")
    pl = {k: {"status": v["status"], "area_px": v["area_px"], "components": len(v["polygons_xy"]), "reason": v["reason"],
              "polygons_xy": v["polygons_xy"], "areas_px": v["areas_px"], "excluded": v["excluded"]}
          for k, v in (ctx.get("placements") or {}).items()}
    goals = {k: None if g is None else {"status": g["status"], "pose": g.get("pose"), "stats": g.get("stats"),
                                        "reach": g.get("reach")} for k, g in ctx["goals"].items()}
    paths = {k: {"status": p["status"], "length_px": p["length_px"], "waypoints_xy": p["waypoints_xy"],
                 "valid": (r["validation"].get(k) or {}).get("valid"),
                 "min_clearance_px": (r["validation"].get(k) or {}).get("min_clearance_px")} for k, p in ctx["paths"].items()}
    ti, ins = r.get("target_inspection") or {}, _step(r, "inspect_target")
    sem = I["semantic"] or {"status": "NOT_RUN", "reasons": ["the semantic step did not run"]}
    drop_ev = ctx.get("dropoff_evals") or {}
    approach_codes = {k: v["status"] for k, v in drop_ev.items()}
    stages = {
        "vlm_proposal": "OK" if (I["vlm"] or {}).get("destination") else (I["vlm"] or {}).get("decision") or "NOT_RUN",
        "semantic_verification": sem["status"],
        "map_alignment": (amap or {}).get("status", "NOT_RUN"),
        "cube_detection": "OK" if ctx.get("target") else (ti.get("status") or "NOT_RUN"),
        "placement": "NOT_RUN" if not pl else ("PLACEMENT_REGION_FOUND" if any(
            v["status"] == "PLACEMENT_REGION_FOUND" for v in pl.values()) else "PLACEMENT_REGION_EMPTY"),
        "approach": "NOT_RUN" if not goals else ",".join(f"{k}:{(g or {}).get('status')}" for k, g in goals.items()),
        "path": "NOT_RUN" if not paths else ",".join(f"{k}:{p['status']}{'' if p['valid'] else '/INVALID'}" for k, p in paths.items()),
    }
    # first blocking stage, in pipeline order
    blocker = None
    if (I["vlm"] or {}).get("destination") is None:
        blocker = ("SEMANTIC_NOT_INFERRED", "vlm_proposal")
    elif sem["status"] == "WITHHELD" or sem["status"] == "SEMANTIC_NOT_VERIFIED":
        blocker = ("SEMANTIC_EVIDENCE_INSUFFICIENT" if sem.get("evidence_failed") else "SEMANTIC_NOT_VERIFIED",
                   "semantic_verification")
    elif stages["map_alignment"] != "ALIGNED":
        blocker = ("MAP_NOT_ALIGNED", "map_alignment")
    elif stages["cube_detection"] != "OK":
        blocker = ("RED_CUBE_NOT_DETECTED", "cube_detection")
    elif stages["placement"] != "PLACEMENT_REGION_FOUND":
        blocker = ("PLACEMENT_REGION_EMPTY", "placement")
    elif not (goals.get("pickup") or {}).get("pose") or not (goals.get("dropoff") or {}).get("pose"):
        blocker = ("ROBOT_APPROACH_NOT_FOUND", "approach")
    elif any(p["status"] != "PATH_FOUND" or not p["valid"] for p in paths.values()) or not paths:
        blocker = ("PATH_NOT_FOUND", "path")
    unverified = []
    start = ctx.get("start") or {}
    if start.get("knowledge") == MANUAL:
        unverified.append("robot start pose: MANUAL prior from the map (the robot was not detected)")
    unverified += ["gripper reach: PROVISIONAL (goal_selection.dropoff.max_dist_px, not measured)",
                   "map: review_status " + run.map_spec["review_status"],
                   "placement surface (height, occupancy): UNMEASURED",
                   "semantic meaning of the adopted region: VLM inference (v0.2 judge is the same model family)"]
    final = blocker[0] if blocker else "PLANNED_IN_IMAGE_SPACE_WITH_UNVERIFIED_INPUTS"
    partial = []
    if blocker and (paths.get("A") or {}).get("status") == "PATH_FOUND":
        partial.append("pickup path A (start -> pickup approach) was planned: v0.2 plans it independently of the destination; "
                       "it is a partial result, not an integration success")
    return {
        "schema": "s2n-integration-0.1", "experiment_id": run.icfg["experiment_id"], "run_id": run.run_id,
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "integration_status": final, "first_blocking_stage": None if blocker is None else blocker[1],
        "partial_results": partial,
        "config_overrides": run.icfg.get("_overrides", []),
        "stage_status": stages,
        "claims": {"robot_task_success": "NOT_CLAIMABLE (image-space planning check only; reach, placement surface and "
                                         "gripper geometry unmeasured; nothing executed; no hardware command exists)",
                   "v02_PLAN_COMPLETE": f"v0.2 overall_status={r['overall_status']} (v0.2 vocabulary; this is NOT a success "
                                        "claim of the integration)"},
        "unverified_inputs": unverified,
        "input": {"image_id": image_spec["id"], "role": image_spec["role"], "file": image_spec["file"],
                  "frame": r["frame"], "instruction": run.command, "instruction_level": run.icfg["instruction_level"]},
        "map_alignment": amap,
        "vlm_proposal": I["vlm"], "verification": {"checks": sorted(I["checks"], key=lambda c: c["id"]),
                                                   "refinement": I["refinement"], "cones": I["cones"]},
        "semantic_decision": sem, "adopted_region": I["adopted"],
        "cube": {"status": stages["cube_detection"], "bbox_xyxy": (ctx.get("target") or {}).get("bbox_xyxy"),
                 "step": ins and {"status": ins["status"], "error": ins["error"]}},
        "placement": pl, "approach": goals, "paths": paths,
        "robot_start": {k: start.get(k) for k in ("xy", "method", "knowledge", "robot_ref")},
        "physical_constraints": {
            "robot_body_inside_yellow_zone": "ASSUMED by the planner: path cells need disc clearance >= robot_radius_px "
                                             f"({run.cfg['planning']['robot_radius_px']} px, assumed) inside the DRAFT map's "
                                             "drivable zone; the real footprint and wheel positions are not modelled",
            "gripper_reaches_object_outside_boundary": "UNVERIFIED (isotropic distance <= PROVISIONAL reach; the single "
                                                       "front gripper's direction and joint limits are not modelled)",
            "collision_with_obstacles": "checked at pixel level against the DRAFT map and detected truck/conveyor only",
            "approach_status": approach_codes},
        "v02_code_unchanged": {"before": v02_before, "after": v02_after, "unchanged": v02_before == v02_after},
        "frozen_vlm_prompt": tools.frozen if hasattr(tools, "frozen") else None,
        "draw": {"vlm_box_px": ((I["vlm"] or {}).get("destination") or {}).get("box_px"),
                 "vlm_label": ((I["vlm"] or {}).get("destination") or {}).get("function_name"),
                 "refined_box_px": ((I["refinement"] or {}).get("chosen") or {}).get("bbox_xyxy"),
                 "adopted_polygon": (I["adopted"] or {}).get("polygon_xy"),
                 "adopted_passed_to_planning": bool(ctx.get("semantic_regions")),
                 "truck_box_px": (I["adopted"] or {}).get("truck_box_px"),
                 "cone_boxes_px": ((I["cones"] or {}).get("traffic_cone") or {}).get("verified", []),
                 "placement_polygons": [p for v in pl.values() for p in v["polygons_xy"]],
                 "target_box_px": (ctx.get("target") or {}).get("bbox_xyxy"),
                 "start_xy": start.get("xy"), "start_manual": start.get("knowledge") == MANUAL,
                 "pickup_goal_xy": ((goals.get("pickup") or {}).get("pose") or {}).get("xy"),
                 "dropoff_goal_xy": ((goals.get("dropoff") or {}).get("pose") or {}).get("xy"),
                 "paths": {k: p["waypoints_xy"] for k, p in paths.items() if p["status"] == "PATH_FOUND"},
                 "drive_zone_polygons": [z["polygon"] for z in run.map_spec["traversable"]] if "nav" in ctx else []},
    }


def v02_fingerprint():
    cfg_hash = {str(p.relative_to(PROJECT)): sha256_file(p) for p in sorted((PROJECT / "config").rglob("*")) if p.is_file()}
    return {"src_and_run_py": code_fingerprint()["sha256"], "config_files": cfg_hash,
            "nla_frozen_prompt": sha256_file(HERE.parent / "natural_language_abstraction" / "prompts_frozen.json"),
            "asr_frozen_prompt": sha256_file(HERE.parent / "autonomous_semantic_reasoning" / "prompts_frozen.json")}


# ------------------------------------------------------------------ CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--image-id", required=True)
    r.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="diagnostic VARIANT of a config value, e.g. verification.max_unrefined_area_ratio=6.0")
    args = ap.parse_args()
    from semantic_task_nav.perception.model_client import ModelWorkerClient
    import visualize
    icfg = apply_overrides(load_icfg(), args.set)
    spec = next((i for i in icfg["images"] if i["id"] == args.image_id), None)
    if spec is None:
        raise SystemExit(f"unknown image id {args.image_id}; known: {[i['id'] for i in icfg['images']]}")
    path = resolve_path(spec["file"])
    if not sha256_file(path).startswith(spec["sha256"]):
        raise SystemExit(f"{spec['id']}: sha256 differs from config.yaml")
    cfg = copy.deepcopy(load_config())
    results = HERE / icfg["results_dir"]
    cfg["app"]["runs_dir"] = str(results)                    # new run folders go here, not into runs/
    cfg["app"]["worker_log_dir"] = str(results / "_worker_logs")
    before = v02_fingerprint()
    frame = ImageSource(path).read()
    client = ModelWorkerClient(cfg["models"], results / "_worker_logs")
    client.start()
    run = None
    try:
        tools = Tools(client, cfg, icfg)
        run = IntegrationRun(cfg, icfg, "LOCAL_ONLY", frame, icfg["instruction"], client, tools,
                             on_event=lambda ev: print(f"[{ev['record']['status']:>9}] {ev['record']['name']:<24} "
                                                       f"{ev['record']['duration_s'] or 0:7.2f}s  "
                                                       f"{(ev['record']['output_summary'] or ev['record']['error'] or '')[:170]}",
                                                       flush=True) if ev["type"] == "step_finished" else None)
        run.run()
    finally:
        client.close()
    after = v02_fingerprint()
    rec = build_record(run, spec, tools, before, after)
    run.writer.write_json("integration_result.json", rec)
    overlay = visualize.render(frame.image, rec)
    run.writer.write_image("integration_overlay.png", overlay)
    files = {str(p.relative_to(run.writer.dir)): sha256_file(p) for p in sorted(run.writer.dir.rglob("*")) if p.is_file()
             and p.name not in ("integration_manifest.json",)}
    run.writer.write_json("integration_manifest.json", {
        "run_id": run.run_id, "experiment_files_sha256": {n: sha256_file(HERE / n) for n in
                                                         ("config.yaml", "integration.py", "verify.py", "visualize.py")},
        "config_overrides": rec["config_overrides"], "v02_unchanged": rec["v02_code_unchanged"]["unchanged"],
        "files_sha256": files})
    print(json.dumps({"run_dir": str(run.writer.dir), "integration_status": rec["integration_status"],
                      "first_blocking_stage": rec["first_blocking_stage"], "stage_status": rec["stage_status"],
                      "v02_unchanged": rec["v02_code_unchanged"]["unchanged"]}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
