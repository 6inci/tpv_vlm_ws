"""Offline checks for the probe (no GPU, no model). Run from the project root:
  /home/binci/miniconda3/envs/stnav_gui/bin/python -m pytest -q experiments/autonomous_semantic_reasoning/test_probe_offline.py
"""
import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe  # noqa: E402

EXP, PROMPTS = probe.load_experiment()
GOOD = {
    "observed_objects": [{"label": "truck", "count": 1, "location_in_image": "top right", "visible_evidence": "cab and cargo bed"}],
    "functional_areas": [{"id": "area_1", "function_name": "vehicle stop area", "bbox_2d": [690, 0, 910, 200],
                          "observed_evidence": ["marking under the truck"], "inference": "vehicles stop here",
                          "confidence": "medium"}],
    "task_analysis": {"place_needed_for_task": "unloading place", "decision": "proposed",
                      "destination_area_id": "area_1", "reason": "next to the truck"},
    "unknown_from_image": ["cube height"], "physical_conditions_to_verify": ["gripper reach"]}


def test_conditions_differ_only_in_prior_block():
    a = probe.render_prompt(PROMPTS, "A", EXP["instruction"])
    for c in ("B", "C1"):
        text = probe.render_prompt(PROMPTS, c, EXP["instruction"])
        prior = PROMPTS["prior"][c].strip("\n")
        assert text.replace("\n\n" + prior, "", 1) == a, f"{c} differs from A outside the prior block"
        assert prior in text and prior not in a


def test_instruction_is_the_users_own_words():
    for c in EXP["conditions"]:
        text = probe.render_prompt(PROMPTS, c, EXP["instruction"])
        assert EXP["instruction"] in text
        assert "트럭" in text and "하역장" in text


# Things that must never reach the model: answer zones, boundary labels, the query phrase chosen on the dev
# image, candidate regions, map / reach information, and any pixel-coordinate-like numbers.
FORBIDDEN = ["Z06", "Z02", "Z0", "yellow", "노란", "parking space", "dashed", "polygon", "reach",
             "drivable", "traversable", "cone", "green", "loading_bay", "loading bay"]


def test_no_answer_or_site_specific_information_in_any_prompt():
    for c in EXP["conditions"]:
        text = probe.render_prompt(PROMPTS, c, EXP["instruction"])
        body = text.replace(EXP["instruction"], "")
        for w in FORBIDDEN:
            assert w.lower() not in body.lower(), f"{c}: forbidden phrase {w!r}"
        numbers = set(re.findall(r"\d+", body))
        # only the 0-1000 coordinate convention, the 5S list numbers, the item limits and the example id
        assert numbers <= {"0", "1000", "1", "2", "3", "4", "5", "12", "6"}, f"{c}: unexpected numbers {numbers}"


def test_validator_accepts_good_and_rejects_bad():
    obj = probe.validate(json.loads(json.dumps(GOOD)), probe.RESPONSE_SCHEMA)
    assert probe.check_response(obj) == []
    bad = json.loads(json.dumps(GOOD))
    bad["functional_areas"][0]["bbox_2d"] = [10, 10, 1500, 20]          # pixel-like coordinates
    with pytest.raises(probe.SchemaError):
        probe.validate(bad, probe.RESPONSE_SCHEMA)
    incons = json.loads(json.dumps(GOOD))
    incons["task_analysis"]["destination_area_id"] = "area_9"
    assert any("not in" in s for s in probe.check_response(incons))
    withheld = json.loads(json.dumps(GOOD))
    withheld["task_analysis"]["decision"] = "withheld"
    assert any("not null" in s for s in probe.check_response(withheld))


class FakeClient:
    """Stands in for ModelWorkerClient; records exactly what would reach the model."""
    def __init__(self, text):
        self.text, self.calls = text, []

    def ensure_loaded(self, name):
        assert name == "qwen"

    def call(self, cmd, **args):
        self.calls.append((cmd, args))
        return {"text": self.text, "n_input_tokens": 1, "n_output_tokens": 1, "hit_max_new_tokens": False,
                "latency_s": 0.0, "image_path": args["image_path"], "image_size": [1920, 1080],
                "max_new_tokens": args["max_new_tokens"], "model": "fake", "commit_hash": None}


def test_only_one_image_and_the_prompt_reach_the_model(tmp_path):
    img = next(i for i in EXP["images"] if i["role"] == "dev")
    client = FakeClient(json.dumps(GOOD))
    res = probe.run_condition(client, EXP, PROMPTS, img, "B", tmp_path / "out")
    (cmd, args), = client.calls
    assert cmd == "qwen_generate" and sorted(args) == ["image_path", "max_new_tokens", "prompt"]
    assert args["prompt"] == probe.render_prompt(PROMPTS, "B", EXP["instruction"])
    assert res["status"] == "OK" and res["structured"] == GOOD
    assert (tmp_path / "out" / "overlay_areas.png").exists() and (tmp_path / "out" / "raw_response.txt").exists()


def test_invalid_reply_is_reported_not_repaired(tmp_path):
    img = next(i for i in EXP["images"] if i["role"] == "dev")
    res = probe.run_condition(FakeClient('{"observed_objects": ['), EXP, PROMPTS, img, "A", tmp_path / "o")
    assert res["status"] == "INVALID_RESPONSE" and res["structured"] is None
    assert (tmp_path / "o" / "raw_response.txt").read_text() == '{"observed_objects": ['


def test_image_registry_is_clean():
    ids = [i["id"] for i in EXP["images"]]
    assert len(ids) == len(set(ids))
    for i in EXP["images"]:
        h = probe.sha256_file(probe.HERE / i["file"])
        assert h == i["sha256"] and not h.startswith(EXP["excluded_image_sha256_prefix"])
    assert {i["role"] for i in EXP["images"]} == {"dev", "eval"}
