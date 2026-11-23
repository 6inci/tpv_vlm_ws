"""Offline checks of the integration adapter (no GPU, no real model). From the project root:
  PYTHONDONTWRITEBYTECODE=1 /home/binci/miniconda3/envs/stnav_gui/bin/python -m pytest -q experiments/semantic_to_navigation_integration/test_integration_offline.py

The end-to-end tests use the v0.2 IMAGE_DEMO inputs (manually supplied objects, labelled manual_test_data) and a scripted
fake for every model call, so they test the PLUMBING (steps, hand-over, stop conditions, provenance), not model quality.
"""
import copy
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import integration as itg  # noqa: E402
import verify  # noqa: E402
import visualize  # noqa: E402
from semantic_task_nav.artifacts import RunWriter  # noqa: E402
from semantic_task_nav.config import load_config  # noqa: E402
from semantic_task_nav.perception.camera import ImageSource  # noqa: E402

ICFG = itg.load_icfg()
IV = ICFG["verification"]
CFG = load_config()
PCFG = CFG["planning"]
REF = itg.PROJECT / "assets" / "reference" / "warehouse_scene_reference_01.jpg"
SIZE = (1920, 1080)
TRUCK = {"object_id": "truck_1", "class_name": "truck", "bbox_xyxy": [1424, 73, 1623, 204], "accepted": True}
CONVEYOR = {"object_id": "conveyor_1", "class_name": "conveyor", "bbox_xyxy": [224, 268, 518, 738], "accepted": True}


def norm(b):
    return [b[0] / 1920 * 1000, b[1] / 1080 * 1000, b[2] / 1920 * 1000, b[3] / 1080 * 1000]


def structured(box_px, decision="proposed", evidence=("white truck parked", "orange cones marking boundaries"),
               inference="the truck is parked here"):
    area = {"id": "area_1", "function_name": "loading dock", "bbox_2d": norm(box_px), "observed_evidence": list(evidence),
            "inference": inference, "confidence": "high"}
    return {"observed_objects": [], "functional_areas": [area],
            "task_analysis": {"inferred_task_purpose": "ship the cube", "info_given_by_instruction": ["a shipping place"],
                              "info_added_by_you": [], "place_needed_for_task": "outbound place", "decision": decision,
                              "destination_area_id": "area_1" if decision == "proposed" else None, "reason": "fake"},
            "unknown_from_image": [], "physical_conditions_to_verify": []}


# ------------------------------------------------------------------ verify.py
def test_image_link_rejects_scene_sized_and_degenerate_boxes():
    assert verify.image_link([1400, 60, 1700, 240], SIZE, IV)["status"] == verify.VERIFIED
    assert verify.image_link([0, 0, 1920, 1080], SIZE, IV)["status"] == verify.FAILED        # the full-frame "zone"
    assert verify.image_link([100, 100, 110, 400], SIZE, IV)["status"] == verify.FAILED      # 10 px wide


def test_evidence_must_name_an_anchor_and_the_anchor_must_exist():
    area = structured([1340, 20, 1760, 300])["functional_areas"][0]
    cited, unv = verify.cited_items(area, IV["evidence_vocabulary"])
    assert {c["class"] for c in cited} == {"truck", "traffic_cone"}
    ok = verify.evidence_existence(cited, unv, [TRUCK], {"traffic_cone": [[1, 1, 5, 5]]})
    assert ok["status"] == verify.VERIFIED
    assert verify.evidence_existence(cited, unv, [], {"traffic_cone": [[1, 1, 5, 5]]})["status"] == verify.FAILED  # cited truck absent
    # cones cited but not confirmed: reported, not blocking
    soft = verify.evidence_existence(cited, unv, [TRUCK], {"traffic_cone": []})
    assert soft["status"] == verify.VERIFIED
    assert any(r["status"] == verify.NOT_CONFIRMED for r in soft["detail"]["cited"])
    # a proposal that never cites the truck is not verified by merely overlapping it
    no_anchor = structured([1340, 20, 1760, 300], evidence=("black floor with lines",),
                           inference="a marked area of the floor")["functional_areas"][0]
    c2, u2 = verify.cited_items(no_anchor, IV["evidence_vocabulary"])
    assert verify.evidence_existence(c2, u2, [TRUCK], {})["status"] == verify.FAILED
    assert u2 == ["black floor with lines"]                                  # listed as UNVERIFIABLE evidence


def test_association_uses_the_v02_thresholds():
    d, r = PCFG["destination"], PCFG["region_grounding"]
    assert verify.association([1380, 40, 1700, 240], TRUCK["bbox_xyxy"], d, r)["status"] == verify.VERIFIED
    assert verify.association([100, 100, 400, 300], TRUCK["bbox_xyxy"], d, r)["status"] == verify.FAILED   # unrelated place
    assert verify.association([1000, 0, 1900, 1000], TRUCK["bbox_xyxy"], d, r)["status"] == verify.FAILED  # scene-sized
    assert verify.association(TRUCK["bbox_xyxy"], TRUCK["bbox_xyxy"], d, r)["status"] == verify.FAILED     # the truck itself


def test_association_zone_clips_to_what_the_anchor_supports():
    z = verify.association_zone(TRUCK["bbox_xyxy"], SIZE, PCFG["region_grounding"])
    assert z[0] == pytest.approx(TRUCK["bbox_xyxy"][0] - 0.6 * 199.12, abs=1.0)           # band_ratio * long side
    clipped = verify.inter_box([1000, 0, 1800, 300], z)
    assert clipped[0] > 1000 and clipped[2] < 1800                                         # far parts are dropped


def test_refined_box_must_be_inside_the_proposal_and_enclose_the_truck():
    d, r, rf = PCFG["destination"], PCFG["region_grounding"], IV["refine"]
    vlm = [1340, 20, 1760, 300]
    assert verify.screen_refined([1400, 60, 1670, 240], vlm, TRUCK["bbox_xyxy"], d, r, rf)[0]
    assert not verify.screen_refined([1100, 20, 1500, 300], vlm, TRUCK["bbox_xyxy"], d, r, rf)[0]      # outside the proposal
    assert not verify.screen_refined([1700, 100, 1759, 200], vlm, TRUCK["bbox_xyxy"], d, r, rf)[0]     # does not enclose the truck


def test_inclusion_catches_swallowed_equipment_conflicting_areas_and_travel_zones():
    layers = {"traversable": np.zeros((1080, 1920), bool), "no_go": np.zeros((1080, 1920), bool)}
    region = [1400, 60, 1670, 240]
    assert verify.inclusion(region, [CONVEYOR], [], layers, IV, SIZE)["status"] == verify.VERIFIED
    swallow = dict(CONVEYOR, bbox_xyxy=[1450, 100, 1500, 150])
    assert verify.inclusion(region, [swallow], [], layers, IV, SIZE)["status"] == verify.FAILED
    other = [{"id": "area_2", "function_name": "storage area", "box_px": [1380, 50, 1700, 250]}]
    assert verify.inclusion(region, [], other, layers, IV, SIZE)["status"] == verify.FAILED
    degenerate = [{"id": "area_9", "function_name": "storage area", "box_px": [0, 0, 1920, 1080]}]
    assert verify.inclusion(region, [], degenerate, layers, IV, SIZE)["status"] == verify.VERIFIED      # ignored, noted
    layers["traversable"][60:240, 1400:1670] = True                                         # region lies in the drive zone
    assert verify.inclusion(region, [], [], layers, IV, SIZE)["status"] == verify.FAILED
    assert verify.inclusion(region, [], [], None, IV, SIZE)["status"] == verify.ESTIMATED   # no aligned map: cannot be VERIFIED


def test_decide_gates_on_required_checks_and_keeps_semantics_when_the_map_is_not_aligned():
    base = [verify.check(i, i, verify.VERIFIED, "t", {}) for i in ("V1", "V2", "V3")]
    v3b = verify.check("V3b", "p", verify.ESTIMATED, "t", {})
    v4 = verify.check("V4", "j", verify.ESTIMATED, "t", {})
    v5v = verify.check("V5", "i", verify.VERIFIED, "t", {})
    v5e = verify.check("V5", "i", verify.ESTIMATED, "t", {})
    assert verify.decide(base + [v3b, v4, v5v], True, "x")[0] == "ADOPTED"
    assert verify.decide(base + [v3b, v4, v5e], False, "x")[0] == "ADOPTED_SEMANTIC_ONLY_MAP_NOT_ALIGNED"
    assert verify.decide(base + [v3b, v4, v5e], True, "x")[0] == "WITHHELD"          # map aligned but check not run properly
    bad = [verify.check("V2", "e", verify.FAILED, "t", {})]
    assert verify.decide([c for c in base if c["id"] != "V2"] + bad + [v3b, v4, v5v], True, "x")[0] == "WITHHELD"
    assert verify.decide(base + [verify.check("V4", "j", verify.FAILED, "t", {}), v3b, v5v], True, "x")[0] == "WITHHELD"


# ------------------------------------------------------------------ adapter plumbing (IMAGE_DEMO + scripted fakes)
class FakeTools:
    frozen = {"ok": True, "message": "fake"}

    def __init__(self, struct, refine=(), cones=(), judge="yes"):
        self.struct, self.refine, self.cones, self.judge_answer = struct, list(refine), list(cones), judge
        self.calls = []

    def propose(self, writer, frame_info):
        self.calls.append("propose")
        decision = self.struct["task_analysis"]["decision"]
        return {"status": "OK", "structured": self.struct, "error": None, "decision": decision, "prompt_sha256": "fake",
                "model_input_audit": {"worker_call_keys": ["image_path", "max_new_tokens", "prompt"], "n_images": 1},
                "call": {"latency_s": 0.0}, "consistency_issues": []}

    def ground(self, image_path, size, query):
        self.calls.append(f"ground:{query}")
        return ([{"bbox_xyxy": list(b), "bbox_norm1000": [0, 0, 1, 1]} for b in self.refine],
                {"raw_response": "fake", "error": None})

    def detect(self, spec, image, image_path, size, region_box, margin_frac, crop_dir):
        self.calls.append("detect")
        return {"n_dino_candidates": len(self.cones), "n_near_proposal": len(self.cones), "checked": [],
                "verified": [list(c) for c in self.cones]}

    def judge(self, image, truck_box, cands, out_dir, margin):
        self.calls.append("judge")
        for c in cands:
            c["vlm_plausible"] = self.judge_answer if c["candidate_id"] == 1 else "no"
            c["vlm_reason"] = "fake"
            c["judgement"] = {"answer": c["vlm_plausible"], "reason": "fake", "input_image": "x", "input_crop": {}, "call": {}}


def run_demo(tmp_path, tools, image=REF):
    cfg = copy.deepcopy(CFG)
    cfg["app"]["runs_dir"] = str(tmp_path / "runs")
    frame = ImageSource(image).read()
    run = itg.IntegrationRun(cfg, ICFG, "IMAGE_DEMO", frame, ICFG["instruction"], None, tools)
    run.run()
    rec = itg.build_record(run, {"id": "TEST", "role": "test", "file": str(image)}, tools, {"x": 1}, {"x": 1})
    return run, rec


GOOD_VLM = [1340, 20, 1760, 300]          # 420 x 280 px around the truck (4.5x the truck box): too wide to use as it is
BAY = [1400, 60, 1670, 240]               # what evidence grounding is scripted to return inside it (1.8x the truck)


def test_instruction_is_the_frozen_l2_sentence_and_the_task_keeps_the_destination_open(tmp_path):
    assert ICFG["instruction"] == "빨간 큐브를 출하할 수 있는 적절한 장소로 옮겨줘."
    run, rec = run_demo(tmp_path, FakeTools(structured(GOOD_VLM), refine=[BAY], cones=[[1500, 250, 1530, 290]]))
    task = run.ctx["task"]
    assert task["target"]["class"] == "cube" and task["target"]["color"] == "red"
    blob = json.dumps(task["destination"]).lower()
    assert "truck" not in blob and "loading" not in blob                      # nothing of the answer enters the task
    assert run.ctx["scene_analysis"]["status"] == "NOT_RUN"


def test_adopted_region_reaches_v02_placement_and_the_vlm_box_is_not_used_as_it_is(tmp_path):
    tools = FakeTools(structured(GOOD_VLM), refine=[BAY], cones=[[1500, 250, 1530, 290]])
    run, rec = run_demo(tmp_path, tools)
    assert rec["semantic_decision"]["status"] == "ADOPTED"
    assert rec["adopted_region"]["basis"] == "refined_by_evidence_grounding"
    assert rec["adopted_region"]["box_px"] == pytest.approx(BAY)                          # tightened, from the VLM's own evidence
    assert rec["adopted_region"]["vlm_box_px"] != rec["adopted_region"]["box_px"]          # raw proposal preserved separately
    region = run.ctx["semantic_regions"][0]
    assert region["verification_status"] == "unverified" and region["source"].startswith("qwen3_vl_free_proposal")
    pl = next(iter(run.ctx["placements"].values()))                                        # v0.2 placement_region ran
    assert pl["semantic_region_id"] == region["region_id"]
    assert {e.get("object_id") for e in pl["excluded"]} >= {"truck_1"}                     # truck footprint excluded
    assert any(c["id"] == "V4" for c in rec["verification"]["checks"])
    assert [c for c in tools.calls if c == "propose"] == ["propose"]
    assert rec["claims"]["robot_task_success"].startswith("NOT_CLAIMABLE")
    assert rec["integration_status"] != "PLAN_COMPLETE"
    img = cv2.imread(str(REF))
    out = visualize.render(img, rec)
    assert out.shape[1] == 1920 and out.shape[0] > 1080                                     # image + legend panel


def test_proposal_that_is_too_wide_and_cannot_be_tightened_is_withheld_not_passed_on(tmp_path):
    run, rec = run_demo(tmp_path, FakeTools(structured(GOOD_VLM), refine=[], cones=[]))
    assert rec["semantic_decision"]["status"] == "SEMANTIC_NOT_VERIFIED"
    assert rec["integration_status"] == "SEMANTIC_NOT_VERIFIED" and rec["first_blocking_stage"] == "semantic_verification"
    assert "semantic_regions" not in run.ctx and not rec["placement"]
    assert "B" not in rec["paths"] and not rec["approach"].get("dropoff")                 # nothing reached the destination chain
    assert rec["paths"]["A"]["status"] == "PATH_FOUND"                                     # pickup path is destination-independent (v0.2)
    assert any("pickup path A" in p for p in rec["partial_results"])                       # ... and is reported as a PARTIAL result
    v3b = next(c for c in rec["verification"]["checks"] if c["id"] == "V3b")
    assert v3b["status"] == verify.FAILED and rec["draw"]["adopted_polygon"] is not None
    assert rec["draw"]["adopted_passed_to_planning"] is False
    visualize.render(cv2.imread(str(REF)), rec)


def test_judge_no_withholds_even_when_geometry_passes(tmp_path):
    run, rec = run_demo(tmp_path, FakeTools(structured(GOOD_VLM), refine=[BAY], judge="no"))
    assert rec["integration_status"] == "SEMANTIC_NOT_VERIFIED" and not rec["placement"]


def test_cited_truck_that_is_not_in_the_image_is_insufficient_evidence(tmp_path, monkeypatch):
    run_cfg = copy.deepcopy(CFG)
    run_cfg["app"]["runs_dir"] = str(tmp_path / "runs")
    frame = ImageSource(REF).read()
    tools = FakeTools(structured(GOOD_VLM), refine=[BAY])
    run = itg.IntegrationRun(run_cfg, ICFG, "IMAGE_DEMO", frame, ICFG["instruction"], None, tools)
    real = run._detect_objects

    def no_truck():
        out = real()
        run.ctx["detections"]["objects"] = [o for o in run.ctx["detections"]["objects"] if o["class_name"] != "truck"]
        return out
    monkeypatch.setattr(run, "_detect_objects", no_truck)
    run.run()
    rec = itg.build_record(run, {"id": "T", "role": "t", "file": str(REF)}, tools, {}, {})
    assert rec["integration_status"] == "SEMANTIC_EVIDENCE_INSUFFICIENT"
    assert "judge" not in tools.calls and "detect" not in tools.calls                     # stopped before any further model call


def test_vlm_withheld_stops_before_verification_and_planning(tmp_path):
    tools = FakeTools(structured(GOOD_VLM, decision="withheld"))
    run, rec = run_demo(tmp_path, tools)
    assert rec["integration_status"] == "SEMANTIC_NOT_INFERRED" and rec["stage_status"]["vlm_proposal"] == "withheld"
    assert tools.calls == ["propose"] and not rec["verification"]["checks"]


def test_planning_steps_are_blocked_when_the_map_is_not_aligned(tmp_path):
    cfg = copy.deepcopy(CFG)
    cfg["app"]["runs_dir"] = str(tmp_path / "runs")
    run = itg.IntegrationRun(cfg, ICFG, "IMAGE_DEMO", ImageSource(REF).read(), ICFG["instruction"], None,
                             FakeTools(structured(GOOD_VLM)))
    run.ctx.update(task={"task_type": "transport_and_place"}, detections={}, semantic_regions=[{"region_id": "r"}])
    miss = run._missing("assess_placement")
    assert miss and miss[0].startswith("nav")                                              # blocked although a region exists
    run.ctx["nav"] = object()
    assert run._missing("assess_placement") == []


def test_misaligned_photo_is_detected_by_v02_alignment_and_nothing_is_corrected():
    from semantic_task_nav.config import resolve_path
    from semantic_task_nav.navigation.map_builder import check_alignment, load_map_spec
    spec = load_map_spec(resolve_path(CFG["app"]["map_file"]))
    ref = cv2.imread(str(REF))
    assert check_alignment(spec, ref, resolve_path)["status"] == "ALIGNED"
    shifted = np.roll(ref, 40, axis=1)
    assert check_alignment(spec, shifted, resolve_path)["status"] == "MISALIGNED"
    new_photo = cv2.imread(str(itg.PROJECT / ICFG["images"][1]["file"]))
    assert check_alignment(spec, new_photo, resolve_path)["status"] == "MISALIGNED"         # the real E1 photo


# ------------------------------------------------------------------ what reaches the model
class CaptureClient:
    def __init__(self):
        self.calls, self.loaded = [], {}          # `loaded` is read by v0.2's TrackedClient, like the real worker client

    def ensure_loaded(self, name):
        assert name == "qwen"

    def call(self, cmd, timeout=None, **args):
        self.calls.append((cmd, args))
        raise RuntimeError("stop after capturing the call")


def test_the_vlm_receives_one_image_and_the_frozen_prompt_only(tmp_path):
    client = CaptureClient()
    tools = itg.Tools(client, copy.deepcopy(CFG), ICFG)                 # also checks the frozen prompt + instruction equality
    assert tools.frozen["ok"]
    writer = RunWriter(tmp_path)
    writer.write_image("input/raw_frame.png", cv2.imread(str(REF)))
    with pytest.raises(RuntimeError):
        tools.propose(writer, {"source_sha256": "abc", "decoded_pixels_sha256": "def"})
    (cmd, args), = client.calls
    assert cmd == "qwen_generate" and sorted(args) == ["image_path", "max_new_tokens", "prompt"]
    assert args["image_path"].endswith("input/raw_frame.png")
    assert args["prompt"] == tools.nla.render(tools.nla_prompts, tools.nla_exp, "L2")
    low = args["prompt"].lower()
    for forbidden in ("z06", "yellow", "parking space", "polygon", "traversable", "map", "detection"):
        assert forbidden not in low.replace(ICFG["instruction"].lower(), "")


def test_semantic_step_model_calls_go_through_the_v02_call_log(tmp_path):
    cfg = copy.deepcopy(CFG)
    cfg["app"]["runs_dir"] = str(tmp_path / "runs")
    from semantic_task_nav.perception.model_client import ModelWorkerError

    class Failing(CaptureClient):                       # v0.2's TrackedClient logs worker errors as ok=False entries
        def call(self, cmd, timeout=None, **args):
            raise ModelWorkerError("TEST", "stop after the call was made")
    raw = Failing()
    tools = itg.Tools(raw, copy.deepcopy(CFG), ICFG)
    assert tools.client is raw
    run = itg.IntegrationRun(cfg, ICFG, "IMAGE_DEMO", ImageSource(REF).read(), ICFG["instruction"], raw, tools)
    assert tools.client is run.client and run.client is not raw                      # v0.2 TrackedClient wrapper
    writer = run.writer
    writer.write_image("input/raw_frame.png", cv2.imread(str(REF)))
    with pytest.raises(ModelWorkerError):
        tools.propose(writer, {"source_sha256": "abc", "decoded_pixels_sha256": "def"})
    logged = [json.loads(l) for l in writer.path("analysis/model_calls.jsonl").read_text().splitlines()]
    assert logged and logged[0]["cmd"] == "qwen_generate" and logged[0]["prompt_sha256"]  # ... and the call is in the log


def test_v02_code_and_frozen_prompts_have_a_fingerprint():
    fp = itg.v02_fingerprint()
    assert len(fp["src_and_run_py"]) == 64 and fp["config_files"] and len(fp["nla_frozen_prompt"]) == 64
