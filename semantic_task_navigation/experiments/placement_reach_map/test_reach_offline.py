"""Offline checks (no GPU, no model). From the project root:
  PYTHONDONTWRITEBYTECODE=1 /home/binci/miniconda3/envs/stnav_gui/bin/python -m pytest -q -p no:cacheprovider experiments/placement_reach_map
(1) gap logic on synthetic input; (2) pins of the latest run (geometry in a PROVISIONAL frame; nothing is claimed about real placement)."""
import json
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import reach_map as R  # noqa: E402

CFG = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
S = 4


def test_gap_is_the_distance_from_the_nominal_body_edge():
    centres = np.zeros((400, 800), bool)
    centres[200, 100:300] = True                                         # body centres on a line y = 50 cm, x = 25..75 cm
    g = R.gap_map(R.body_union(centres, (35, 24), S), S)
    assert g[200, 200] == 0                                              # under the body
    assert abs(g[200 - 48 - 40, 200] - 10) <= 0.5                        # 10 cm above the body edge (edge is 12 cm from the centre line)
    assert abs(g[200, 300 + 70 + 80] - 20) <= 0.5                        # 20 cm beyond the last body's front edge (17.5 cm from the last centre)


def test_body_is_never_drawn_larger_than_nominal():
    c = np.zeros((200, 200), bool)
    c[100, 100] = True
    u = R.body_union(c, (35, 24), S)
    ys, xs = np.nonzero(u)
    assert (xs.max() - xs.min() + 1) / S <= 35 + 1e-9 and (ys.max() - ys.min() + 1) / S <= 24 + 1e-9


def test_r_sweep_is_monotone_and_reports_zero_when_nothing_is_within_r():
    cand = np.zeros((100, 100), bool)
    cand[10:20, 10:20] = True
    gap = np.full((100, 100), 30.0)
    sw = R.r_sweep(gap, cand, [0, 10, 30, 40], S)
    assert [x["area_cm2"] for x in sw] == [0.0, 0.0, 6.2, 6.2] and sw[-1]["fraction_of_candidate"] == 1.0


def test_start_where_the_body_does_not_fit_gives_no_reachable_set():
    class Nav:
        footprint_fits_full = np.zeros((50, 50), bool)
    assert R.reachable_centres(Nav(), [10, 10]) is None


def test_latest_run_pins():
    runs = sorted((HERE / CFG["results_dir"]).glob("*_reach"))
    assert runs, "run reach_map.py first"
    r = json.loads((runs[-1] / "result.json").read_text(encoding="utf-8"))
    assert r["model_calls"] == 0 and r["v02_code_before"] == r["v02_code_after"] and r["v02_code_before"].startswith("254cfb31da7f")
    assert r["claims"]["placement_success"] == "NOT_CLAIMABLE" and r["claims"]["gripper_reach"].startswith("NOT ASSUMED")
    assert r["state"].startswith("PLACEMENT_UNVERIFIED") and r["placement_point"]["state"] == "PLACEMENT_UNVERIFIED"
    pc = r["placement_candidate"]
    assert pc["area_cm2"] > 0 and pc["overlap_with_drivable_cm2"] == 0.0 and "GEOMETRY_ONLY" in pc["status"]
    for hs in r["heading_sets"].values():
        a = [x["area_cm2"] for x in hs["R_sweep"]]
        assert a == sorted(a) and a[0] == 0.0 and a[-1] == pc["area_cm2"]
        assert hs["min_gap_cm"] > 0                                       # the body never touches the candidate floor
    b = r["path_B_to_placement_stop"]
    assert b["generated"] and b["whole_body_sweep"]["ok"] and b["layers"]["clear"] and b["path"]["waypoints_cm"][0] == r["start_of_path_B_cm"]
    assert all((runs[-1] / "figures" / f).exists() for f in r["figures"]) and len(r["figures"]) == 4
