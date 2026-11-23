"""STEP 2 offline checks (no model, no GPU, no ROS).
  /home/binci/miniconda3/envs/stnav_gui/bin/python -m pytest -q -p no:cacheprovider experiments/navigation_handoff/tests
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import coordinate_transform as CT  # noqa: E402
import map_export as ME  # noqa: E402


def _frame():
    return CT.MapFrame(np.eye(3), [0, 210])          # image px == floor cm, map origin at floor (0, 210)


def test_floor_map_roundtrip_handedness_and_yaw():
    f = _frame()
    assert np.allclose(f.floor_to_map([[0, 210], [315, 210], [315, 0]]), [[0, 0], [3.15, 0], [3.15, 2.10]])
    pts = np.array([[12.3, 45.6], [300.0, 7.5]])
    assert np.allclose(f.map_to_floor(f.floor_to_map(pts)), pts)
    ex, ey = f.floor_to_map([[1, 210]])[0], f.floor_to_map([[0, 209]])[0]     # floor +x, and one cm away from the camera
    assert ex[0] * ey[1] - ex[1] * ey[0] > 0                                  # right-handed in the map frame
    assert CT.MapFrame.yaw_floor_to_map(90) == -90.0 and CT.MapFrame.yaw_floor_to_map(-90) == 90.0
    assert CT.MapFrame.yaw_floor_to_map(180) == 180.0 and CT.MapFrame.yaw_floor_to_map(0) == 0.0


def test_grid_cell_centres_follow_nav2_convention():
    f = _frame()
    g = CT.MapGrid(f, [-10, -10, 325, 220], 0.02)
    assert (g.rows, g.cols) == (115, 168) and np.allclose(g.origin_m[:2], [-0.10, -0.10])
    rr, cc = np.mgrid[0:g.rows, 0:g.cols]
    xf, yf = g.cell_center_floor(rr, cc)
    a = f.floor_to_map(np.c_[xf.ravel(), yf.ravel()])
    bx, by = g.cell_center_map(rr, cc)
    assert np.abs(a - np.c_[bx.ravel(), by.ravel()]).max() < 1e-12
    r, c = g.map_to_cell(bx.ravel(), by.ravel())
    assert (r == rr.ravel()).all() and (c == cc.ravel()).all()
    assert g.cell_center_map(0, 0)[1] > g.cell_center_map(g.rows - 1, 0)[1]   # row 0 is the top (largest y)


def test_rasterize_is_conservative():
    f = _frame()
    g = CT.MapGrid(f, [-4, -4, 24, 14], 0.02)                                 # 2 cm cells, workspace 20 x 10 cm
    zones = {"traversable": [{"id": "T", "polygon": [[0, 0], [15, 0], [15, 10], [0, 10]]}],
             "static_obstacles": [{"id": "O", "polygon": [[4, 4], [6, 4], [6, 6], [4, 6]]}], "no_go": []}
    code, _ = ME.rasterize(zones, g, (20, 10), 4)
    cell = lambda x, y: code[int((y + 4) // 2), int((x + 4) // 2)]  # noqa: E731
    assert cell(1, 1) == ME.FREE                    # fully inside the traversable zone
    assert cell(5, 5) == ME.STATIC_OBSTACLE
    assert cell(15, 1) == ME.NOT_DRIVABLE           # straddles the zone edge (x 14..16): not free
    assert cell(18, 5) == ME.NOT_DRIVABLE           # inside the workspace, outside the zone
    assert cell(-3, 5) == ME.UNKNOWN and cell(22, 5) == ME.UNKNOWN
    assert cell(21, 5) == ME.NOT_DRIVABLE           # cell 20..22 mixes workspace and outside: occupied, not unknown


def test_pgm_roundtrip_and_nav2_trinary(tmp_path):
    arr = np.array([[254, 205, 0], [0, 254, 205]], np.uint8)
    ME.write_pgm(tmp_path / "m.pgm", arr, "test")
    back, hdr = ME.read_pgm(tmp_path / "m.pgm")
    assert np.array_equal(back, arr) and hdr["magic"] == "P5" and (hdr["width"], hdr["height"], hdr["maxval"]) == (3, 2, 255)
    assert ME.nav2_trinary([254, 205, 0], 0, 0.65, 0.196).tolist() == [0, -1, 100]


def test_photos_resolve_by_hash():
    cfg = CT.load_config()
    if not Path(cfg["photo_dir"]).exists():
        pytest.skip("photo directory not available")
    photos = CT.resolve_photos(cfg)
    assert len({p["path"] for p in photos}) == 8
    assert next(p for p in photos if p["name"] == cfg["reference_photo"])["earlier_id"] == "U25070552"
