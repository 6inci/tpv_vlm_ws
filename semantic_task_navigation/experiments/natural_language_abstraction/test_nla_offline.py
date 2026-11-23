"""Offline checks (no GPU, no model). From the project root:
  PYTHONDONTWRITEBYTECODE=1 /home/binci/miniconda3/envs/stnav_gui/bin/python -m pytest -q experiments/natural_language_abstraction/test_nla_offline.py
"""
import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nla_probe as nla  # noqa: E402

EXP, PROMPTS = nla.load_experiment()
asr = nla.asr
LEVELS = list(EXP["instruction_levels"])

GOOD = {
    "observed_objects": [{"label": "truck", "count": 1, "location_in_image": "top right", "visible_evidence": "cab and bed"}],
    "functional_areas": [{"id": "area_1", "function_name": "vehicle stop area", "bbox_2d": [690, 0, 910, 200],
                          "observed_evidence": ["marking under the truck"], "inference": "vehicles stop here",
                          "confidence": "medium"}],
    "task_analysis": {"inferred_task_purpose": "move the cube out", "info_given_by_instruction": ["a place for shipping"],
                      "info_added_by_you": [{"statement": "trucks carry goods out", "basis": "general_knowledge"}],
                      "place_needed_for_task": "outbound place", "decision": "proposed",
                      "destination_area_id": "area_1", "reason": "next to the truck"},
    "unknown_from_image": ["cube height"], "physical_conditions_to_verify": ["gripper reach"]}


def test_instructions_are_exactly_the_users_three_sentences():
    assert EXP["instruction_levels"] == {
        "L1": "빨간 큐브를 트럭이 있는 하역장으로 옮겨줘.",
        "L2": "빨간 큐브를 출하할 수 있는 적절한 장소로 옮겨줘.",
        "L3": "생산된 빨간 큐브를 다음 물류 단계로 전달해 줘."}
    assert EXP["condition"] == "A" and PROMPTS["prior"]["A"] == ""


def test_levels_differ_only_in_the_instruction():
    texts = {lv: nla.render(PROMPTS, EXP, lv) for lv in LEVELS}
    masked = {lv: t.replace(EXP["instruction_levels"][lv], "<INSTRUCTION>") for lv, t in texts.items()}
    assert len(set(masked.values())) == 1, "prompts differ outside the instruction"
    assert len(set(texts.values())) == 3
    for lv, t in texts.items():
        assert t.count(EXP["instruction_levels"][lv]) == 1


# The head/tail must not hand the model a destination concept; only the instruction may contain such words.
FORBIDDEN = ["truck", "loading", "unloading", "shipping", "shipment", "dispatch", "dock", "yard", "bay", "parking",
             "yellow", "green", "cone", "dashed", "polygon", "reach", "Z0", "drivable", "5S", "storage", "charging"]


def test_prompt_body_names_no_destination_or_site_specific_thing():
    for lv in LEVELS:
        body = nla.render(PROMPTS, EXP, lv).replace(EXP["instruction_levels"][lv], "")
        for w in FORBIDDEN:
            assert w.lower() not in body.lower(), f"{lv}: forbidden word {w!r} in prompt body"
        # 0-1000 coordinate convention, "bbox_2d", "area_1", and the item limits 12 / 6 -- nothing else
        assert set(re.findall(r"\d+", body)) <= {"0", "1000", "1", "2", "12", "6"}, "unexpected numbers in the prompt body"


def test_only_the_intended_parts_differ_from_the_frozen_parent_prompt():
    parent = asr.load_yaml(nla.PARENT / "prompts.yaml")
    p_head, n_head = set(parent["head"].splitlines()), set(PROMPTS["head"].splitlines())
    assert len(p_head - n_head) == 2 and len(n_head - p_head) == 3, (p_head - n_head, n_head - p_head)
    # the object/area part of the output format is byte-identical to the parent's
    def block(t):
        return t[t.index('"observed_objects"'):t.index('"task_analysis"')]
    assert block(parent["tail"]) == block(PROMPTS["tail"])
    assert PROMPTS["prior"]["A"] == parent["prior"]["A"] == ""


def test_schema_requires_the_new_fields_and_checks_basis():
    obj = asr.validate(json.loads(json.dumps(GOOD)), nla.SCHEMA)
    assert asr.check_response(obj) == []
    for field in ("inferred_task_purpose", "info_given_by_instruction", "info_added_by_you"):
        bad = json.loads(json.dumps(GOOD))
        del bad["task_analysis"][field]
        with pytest.raises(asr.SchemaError):
            asr.validate(bad, nla.SCHEMA)
    bad = json.loads(json.dumps(GOOD))
    bad["task_analysis"]["info_added_by_you"][0]["basis"] = "magic"
    with pytest.raises(asr.SchemaError):
        asr.validate(bad, nla.SCHEMA)
    zero = json.loads(json.dumps(GOOD))
    zero["observed_objects"].append({"label": "red cube", "count": 0, "location_in_image": "not visible",
                                     "visible_evidence": "looked for it, not seen"})
    asr.validate(zero, nla.SCHEMA)                                         # count 0 = "looked for, not seen" is allowed
    zero["observed_objects"][1]["count"] = -1
    with pytest.raises(asr.SchemaError):
        asr.validate(zero, nla.SCHEMA)
    withheld = json.loads(json.dumps(GOOD))
    withheld["task_analysis"].update(decision="withheld", destination_area_id=None)
    assert asr.check_response(asr.validate(withheld, nla.SCHEMA)) == []   # "withheld" is a valid outcome


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


@pytest.mark.parametrize("level", LEVELS)
def test_one_image_and_one_prompt_reach_the_model(tmp_path, level):
    img = next(i for i in EXP["images"] if i["role"] == "smoke")
    client = FakeClient(json.dumps(GOOD))
    res = nla.run_level(client, EXP, PROMPTS, img, level, tmp_path / "out")
    (cmd, args), = client.calls
    assert cmd == "qwen_generate" and sorted(args) == ["image_path", "max_new_tokens", "prompt"]
    assert args["prompt"] == nla.render(PROMPTS, EXP, level)
    assert res["status"] == "OK" and res["structured"] == GOOD and res["instruction"] == EXP["instruction_levels"][level]
    assert res["condition"] == level
    for f in ("prompt.txt", "raw_response.txt", "structured.json", "result.json", "overlay_areas.png"):
        assert (tmp_path / "out" / f).exists()


def test_invalid_reply_is_reported_not_repaired(tmp_path):
    img = next(i for i in EXP["images"] if i["role"] == "smoke")
    res = nla.run_level(FakeClient('{"observed_objects": ['), EXP, PROMPTS, img, "L3", tmp_path / "o")
    assert res["status"] == "INVALID_RESPONSE" and res["structured"] is None
    assert (tmp_path / "o" / "raw_response.txt").read_text() == '{"observed_objects": ['


def test_images_are_the_parents_and_the_parent_is_untouched():
    parent_cfg = asr.load_yaml(nla.PARENT / "config.yaml")
    parent_hash = {i["id"]: i["sha256"] for i in parent_cfg["images"]}
    for i in EXP["images"]:
        h = asr.sha256_file(nla.HERE / i["file"])
        assert h == i["sha256"] == parent_hash[i["id"]] and not h.startswith(EXP["excluded_image_sha256_prefix"])
    assert {i["id"] for i in EXP["images"] if i["role"] == "exploratory"} == {"E1", "E2"}
    ok, msg = asr.check_frozen(*asr.load_experiment())          # parent's frozen v3 prompts still match
    assert ok, msg
