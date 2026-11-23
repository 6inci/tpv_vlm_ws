"""Offline checks for run_experiment.py (synthetic inputs; no model, no GPU).

  PYTHONDONTWRITEBYTECODE=1 /home/binci/miniconda3/envs/stnav_gui/bin/python -m pytest -q -p no:cacheprovider experiments/task_conditioned_target_grounding/test_tctg_offline.py
"""
import inspect
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_experiment as R  # noqa: E402

CFG = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))


def test_no_answer_in_prompts_instructions_or_rules():
    forbidden = ["pallet", "팔레트", "grate", "격자", "1593", "422,", "회색"]
    texts = [CFG["interpretation"]["template"], CFG["interpretation"]["neutral"], CFG["roles"]["prompt"], CFG["selection"]["template"]] + \
            [v["text"] for v in CFG["instructions"].values()] + [str(CFG["regions"])]
    for t in texts:
        assert not [w for w in forbidden if w in t.lower()], t[:80]
    for fn in (R.decide, R.numbered_set, R.role_queue, R.bucket, R.interp_phrases, R.map_boxes):
        assert "evaluation_reference" not in inspect.getsource(fn)
    src = inspect.getsource(R.main)
    assert src.index("evaluate(") > src.index("target_selection.json")


def test_map_boxes_full_and_crop():
    obj = {"source_region": {"bbox_2d": [0, 0, 500, 500]}, "source_object": {"name_en": "x", "bbox_2d": None},
           "destination_region": {"bbox_2d": [500, 500, 1000, 1000]}, "destination_target_candidates": [{"name_en": "y", "bbox_2d": [0, 0, 1000, 1000]}]}
    R.map_boxes(obj, [1000, 0, 1400, 200])
    assert obj["source_region"]["bbox_image_px"] == [1000.0, 0.0, 1200.0, 100.0] and obj["source_object"]["bbox_image_px"] is None
    assert obj["destination_target_candidates"][0]["bbox_image_px"] == [1000.0, 0.0, 1400.0, 200.0]


def test_interp_phrases_cover_all_name_fields_only():
    parsed = {"A1": {"source_object": {"name_en": "Red Cube"}, "destination_target_candidates": [{"name_en": "Truck Bed."}, {"name_en": "red cube"}]},
              "A3": {"objects": [{"name_en": "cone"}], "support_candidates": [{"name_en": "work table"}]}, "T4": None}
    phrases, org = R.interp_phrases(parsed, 10)
    assert phrases == ["red cube", "truck bed", "cone", "work table"] and [o["call"] for o in org["red cube"]] == ["A1", "A1"]


def _c(i, box, score):
    return {"id": i, "bbox_xyxy": box, "score": score}


def test_role_queue_budget_per_bucket_and_nothing_dropped():
    rects = {"SRC": [0, 0, 100, 100], "DST": [200, 0, 300, 100]}
    merged = [_c("a", [10, 10, 20, 20], .5), _c("b", [30, 30, 40, 40], .9), _c("c", [210, 10, 220, 20], .1), _c("d", [500, 500, 510, 510], .3)]
    q, beyond = R.role_queue(merged, rects, {"DST": 1, "SRC": 1, "OTHER": 0})
    assert [(c["id"], b) for c, b in q] == [("c", "DST"), ("b", "SRC")] and sorted(c["id"] for c, _ in beyond) == ["a", "d"]


def _a(i, sup, load):
    return {"id": i, "bbox_xyxy": [0, 0, 1, 1], "score": .3, "role": {"object_type": "t", "is_support_surface": sup, "load_state": load}}


def test_numbered_set_rule():
    an = [_a("s", "yes", "empty"), _a("n", "no", "not_applicable"), _a("o", "no", "occupied"), _a("u", "uncertain", "uncertain"), {"id": "x", "role": None}]
    assert [a["id"] for a in R.numbered_set(an, 10)] == ["s", "o", "u"] and len(R.numbered_set(an, 2)) == 2


def test_decide_rules():
    num = [_a("empty", "yes", "empty"), _a("full", "yes", "occupied"), _a("notsup", "no", "not_applicable")]
    d = R.decide({"source_id": 2, "destination_id": 1}, num)
    assert (d["source_status"], d["source"], d["destination_status"], d["destination"]) == ("SELECTED", "full", "SELECTED", "empty")
    assert R.decide({"source_id": None, "destination_id": 2}, num)["destination_status"] == "UNVERIFIED"      # occupied cannot receive
    assert R.decide({"source_id": 1, "destination_id": 3}, num)["destination_status"] == "UNVERIFIED"         # not a support
    assert R.decide({"source_id": 9, "destination_id": None}, num)["source_status"] == "AMBIGUOUS"
    assert R.decide({"source_id": 1, "destination_id": 7}, num)["destination_status"] == "AMBIGUOUS"
    assert R.decide(None, num)["destination_status"] == "AMBIGUOUS" and R.decide({"source_id": 1, "destination_id": 1}, [])["destination_status"] == "NOT_FOUND"
