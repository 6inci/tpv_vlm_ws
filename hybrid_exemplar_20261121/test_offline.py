#!/usr/bin/env python3
"""No-model checks of the hybrid pipeline's pure functions and its separation from the answer file.
  /home/binci/miniconda3/envs/stnav_gui/bin/python -B test_offline.py
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hybrid_pipeline as HP  # noqa: E402


def c(i, b):
    return {"id": i, "bbox_xyxy": b}


def main():
    D = HP.CFG["decision"]
    cube, pad, cone = c("cube", [100, 100, 117, 117]), c("pad", [92, 92, 140, 138]), c("cone", [400, 400, 420, 425])
    # the pallet under the cube is not a competitor of the cube; a far cone is
    assert not HP.distinct(cube["bbox_xyxy"], pad["bbox_xyxy"], D["distinct_iou"])
    assert HP.distinct(cube["bbox_xyxy"], cone["bbox_xyxy"], D["distinct_iou"])
    args = (D["min_margin"], D["nms_iou"], D["distinct_iou"])
    assert HP.decide([cube, pad, cone], {"cube": 0.93, "pad": 0.92, "cone": 0.86}, 0.88, *args)["status"] == "SELECTED"
    assert HP.decide([cube, cone], {"cube": 0.90, "cone": 0.89}, 0.88, *args)["status"] == "AMBIGUOUS"
    assert HP.decide([cube, cone], {"cube": 0.87, "cone": 0.80}, 0.88, *args)["status"] == "NOT_FOUND"     # no cube in the photo -> hold
    assert HP.decide([], {}, 0.88, *args)["status"] == "NOT_FOUND"
    # exemplar score = mean over exemplars and views of the view-matched cosine
    e = np.eye(2, 4)
    assert abs(HP.exemplar_scores({"x": e}, [e, -e])["x"]) < 1e-9 and abs(HP.exemplar_scores({"x": e}, [e])["x"] - 1) < 1e-9
    # bank: leave-one-photo-out and same-slot drop
    bank = [{"photo_id": "P1", "slot": [0, 0, 50, 50]}, {"photo_id": "P2", "slot": [0, 0, 50, 50]}, {"photo_id": "P3", "slot": [500, 500, 550, 550]}]
    keep, drop = HP.bank_for(bank, [{"kind": "empty_pad", "box": [2, 2, 50, 50]}], "P1", 0.3)
    assert [k["photo_id"] for k in keep] == ["P3"] and [d["photo_id"] for d in drop] == ["P2"]
    # graspable list
    assert HP.source_classes("any object", ["cube"]) == ["cube"] and HP.source_classes("other", ["cube"]) == []
    # slot reply schema: valid reply passes, a relation outside the list is rejected
    ok = '{"object": "any object", "object_words": "물체", "source_landmark": "truck_loading_bay", "source_relation": "near", ' \
         '"destination_kind": "near_landmark", "destination_landmark": "conveyor", "destination_relation": "near", "destination_words": "컨베이어 벨트 옆"}'
    assert HP.P.parse_reply(ok, HP.SLOT_SCHEMA)[0] is not None
    assert HP.P.parse_reply(ok.replace('"source_relation": "near"', '"source_relation": "in"'), HP.SLOT_SCHEMA)[0] is None
    # the pipeline never opens the answer file
    assert "ground_truth" not in (HERE / "hybrid_pipeline.py").read_text(encoding="utf-8")
    print("test_offline: all checks passed")


if __name__ == "__main__":
    main()
