"""Test C (command -> structure) and Test K (no fabricated plan on bad input)."""
import json

import pytest

from conftest import EXAMPLE_COMMAND
from semantic_task_nav.agent.task_parser import (ACTION_TEMPLATES, TaskParseError, build_task, parse_rule_based,
                                                 parse_with_llm)
from semantic_task_nav.config import load_config

PROMPT = load_config()["models"]["prompts"]["task_parse"]
OK = {"status": "ok", "task_type": "transport_and_place",
      "target": {"class": "cube", "color": "red", "count": 1, "source_reference": "conveyor belt",
                 "source_relation": "produced_by"},
      "destination": {"region_type": "loading dock", "reference_object": "truck"}, "reason": "x"}


def llm(obj_or_text):
    text = obj_or_text if isinstance(obj_or_text, str) else json.dumps(obj_or_text)
    return lambda prompt: text


def parse(reply):
    return parse_with_llm(EXAMPLE_COMMAND, llm(reply), PROMPT, "fake_llm", "fake")


def test_rule_based_example_command():
    t = parse_rule_based(EXAMPLE_COMMAND)
    assert t["task_type"] == "transport_and_place"
    assert t["target"] == {"class": "cube", "color": "red", "source_reference": "conveyor", "source_relation": "produced_by"}
    assert t["destination"] == {"region_type": "loading_area", "reference_object": "truck"}
    assert t["actions"] == ACTION_TEMPLATES["transport_and_place"]
    assert t["parser"]["backend"] == "rule_based_demo"


def test_rule_based_navigation_command():
    t = parse_rule_based("트럭이 있는 하역장으로 이동해.")
    assert t["task_type"] == "navigate_to_region" and t["target"] is None


def test_llm_output_is_normalised_and_raw_kept():
    t = parse("```json\n" + json.dumps(OK) + "\n```")
    assert t["target"]["source_reference"] == "conveyor"
    assert t["destination"]["region_type"] == "loading_area"
    assert t["parser"]["raw_response"].startswith("```json")


def test_model_cannot_choose_actions():
    t = parse({**OK, "actions": ["run_python", "delete_files"]})
    assert t["actions"] == ACTION_TEMPLATES["transport_and_place"]


@pytest.mark.parametrize("reply,code", [
    ("not json at all", "INVALID_JSON"),
    ('{"status": "ok", "task_type": "transport_and_place"', "INVALID_JSON"),
    ({k: v for k, v in OK.items() if k != "destination"}, "SCHEMA_INVALID"),
    ({**OK, "status": "unsupported", "reason": "rotate"}, "UNSUPPORTED_TASK"),
    ({**OK, "status": "ambiguous"}, "AMBIGUOUS_COMMAND"),
    ({**OK, "status": "missing_information"}, "MISSING_INFORMATION"),
    ({**OK, "target": {**OK["target"], "count": 2}}, "MULTIPLE_TARGETS"),
    ({**OK, "target": {**OK["target"], "class": "banana"}}, "UNSUPPORTED_TARGET"),
    ({**OK, "target": {**OK["target"], "color": "blue"}}, "UNSUPPORTED_TARGET"),
    ({**OK, "destination": {"region_type": "kitchen", "reference_object": None}}, "UNSUPPORTED_DESTINATION"),
    ({**OK, "destination": {"region_type": "loading_area", "reference_object": None}}, "UNSUPPORTED_DESTINATION"),
    ({**OK, "target": None}, "MISSING_INFORMATION"),
])
def test_invalid_outputs_are_rejected(reply, code):
    with pytest.raises(TaskParseError) as e:
        parse(reply)
    assert e.value.code == code


def test_rule_based_rejections():
    for cmd, code in (("빨간 큐브를 회전시켜", "UNSUPPORTED_TASK"),
                      ("빨간 큐브와 파란 큐브를 트럭 하역장에 놓아줘", "MULTIPLE_TARGETS"),
                      ("빨간 큐브를 옮겨줘", "MISSING_INFORMATION")):
        with pytest.raises(TaskParseError) as e:
            parse_rule_based(cmd)
        assert e.value.code == code, cmd


def test_build_task_rejects_status_ok_without_type():
    with pytest.raises(TaskParseError):
        build_task({**OK, "task_type": None}, "x", "raw")
