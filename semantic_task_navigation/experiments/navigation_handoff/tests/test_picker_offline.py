"""Approved-position files and the position picker GUI (offscreen). No model, no GPU, no ROS.
  /home/binci/miniconda3/envs/stnav_gui/bin/python -m pytest -q -p no:cacheprovider experiments/navigation_handoff/tests
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import coordinate_transform as CT  # noqa: E402
import object_positions as OP  # noqa: E402

ST = {"A_obj": [442.5, 424.5], "A_tgt": [442.5, 424.5], "A_tgt_src": "SAME_AS_OBJECT_CLICK", "B_obj": [1616.0, 270.0], "B_tgt": [1618.0, 268.0],
      "B_tgt_src": "USER_SET", "robot": [1700.0, 700.0], "robot_src": "USER_CORRECTED", "yaw": 180.0, "yaw_src": "USER_SET_BUTTON"}


def _cfg(tmp_path):
    cfg = CT.load_config()
    cfg["approved_positions_dir"] = str(tmp_path)
    return cfg


def test_approved_roundtrip_refusal_and_immutable_history(tmp_path):
    cfg = _cfg(tmp_path)
    photo = {**cfg["photos"][4], "path": "/nowhere/5calibration.png"}
    frame = CT.MapFrame(np.eye(3), [0, 210])
    doc = OP.build_approved(photo, ST, frame, None, "2026-11-20T10:00:00+09:00")
    assert doc["A_cube"]["grasp_target_status"] == "UNVERIFIED" and doc["B_placement"]["placement_target_source"] == "USER_SET"
    cur, hist = OP.save_approved(cfg, doc, np.zeros((8, 8, 3), np.uint8))
    d, _ = OP.load_approved(cfg, photo["name"])
    assert d["B_placement"]["placement_target_image_px"] == [1618.0, 268.0] and d["robot"]["yaw_map_deg"] == 180.0
    with pytest.raises(FileExistsError):                                  # a history entry is never overwritten
        OP.save_approved(cfg, doc, np.zeros((8, 8, 3), np.uint8))
    bad = dict(d, status="CANDIDATE")
    cur.write_text(yaml.safe_dump(bad))
    with pytest.raises(ValueError):
        OP.load_approved(cfg, photo["name"])
    bad = dict(d, photo_sha256="0" * 64)
    cur.write_text(yaml.safe_dump(bad))
    with pytest.raises(ValueError):
        OP.load_approved(cfg, photo["name"])
    with pytest.raises(FileNotFoundError):
        OP.load_approved(cfg, "photo_1")


def test_yaw_from_click_uses_the_map_frame():
    frame = CT.MapFrame(np.eye(3), [0, 210])                            # image px == floor cm
    assert OP.yaw_from_points(frame, [100, 100], [150, 100]) == 0.0      # image right = +x_map
    assert OP.yaw_from_points(frame, [100, 100], [100, 50]) == 90.0      # image up = away from the camera = +y_map
    assert OP.yaw_from_points(frame, [100, 100], [50, 100]) == 180.0


def test_picker_gui(tmp_path, monkeypatch):
    """tkinter GUI logic. Needs an X display: run the suite under xvfb-run so no window appears on the user's screen."""
    cfg = _cfg(tmp_path)
    if not Path(cfg["photo_dir"]).exists() or not os.environ.get("DISPLAY"):
        pytest.skip("photo directory or X display not available")
    import tkinter as tk
    import position_picker as PK
    monkeypatch.setattr(PK.CT, "load_config", lambda *a, **k: cfg)
    root = tk.Tk()
    try:
        w = PK.Picker(root)
        root.update()
        w.select_index(4)                                                  # photo_5: has a stored automatic robot estimate
        root.update()
        assert w.st["robot_src"] == "AUTO_STORED_DETECTION" and w.st["yaw"] is None and w.btn_approve.instate(["disabled"])
        w.mode.set("A_obj")
        w.on_click(440.2, 425.3)
        w.mode.set("B_obj")
        w.on_click(1616.0, 270.0)
        w.set_yaw(180.0, "USER_SET_BUTTON")
        root.update()
        assert w.st["A_tgt"] == [440.2, 425.3] and w.st["A_tgt_src"] == "SAME_AS_OBJECT_CLICK"
        assert w.btn_approve.instate(["!disabled"]) and w.btn_plan.instate(["disabled"])   # nothing is planned before approval
        w.mode.set("A_tgt")
        w.on_click(441.0, 426.0)
        assert w.st["A_tgt_src"] == "USER_SET" and w.st["A_obj"] == [440.2, 425.3]
        cx, cy = w.view.to_canvas(1000.0, 500.0)                           # canvas <-> image round trip
        assert np.allclose(w.view.to_img(cx, cy), (1000.0, 500.0))
        monkeypatch.setattr(PK.messagebox, "askyesno", lambda *a, **k: True)
        w.approve()
        assert w.st["approved"] and w.btn_plan.instate(["!disabled"])
        assert OP.load_approved(cfg, "photo_5")[0]["A_cube"]["grasp_target_source"] == "USER_SET"
        w.select_index(0)                                                  # photo_1: no stored robot estimate
        assert w.st["robot"] is None and w.st["robot_src"] is None
    finally:
        root.destroy()
