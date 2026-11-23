import copy
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from semantic_task_nav.config import load_config  # noqa: E402

REF_IMAGE = ROOT / "assets" / "reference" / "warehouse_scene_reference_01.jpg"
EXAMPLE_COMMAND = "컨베이어 벨트에서 생산된 빨간색 큐브를 트럭이 위치한 하역장에 놓아줘."


def pytest_configure(config):
    config.addinivalue_line("markers", "models: needs the qwen_vl model worker + GPU (run with STNAV_RUN_MODEL_TESTS=1)")


def pytest_collection_modifyitems(config, items):
    if os.environ.get("STNAV_RUN_MODEL_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="real-model test: set STNAV_RUN_MODEL_TESTS=1")
    for it in items:
        if "models" in it.keywords:
            it.add_marker(skip)


@pytest.fixture
def cfg(tmp_path):
    c = load_config()
    c["app"]["runs_dir"] = str(tmp_path / "runs")
    c["app"]["worker_log_dir"] = str(tmp_path / "worker_logs")
    return c


@pytest.fixture
def pcfg():
    c = copy.deepcopy(load_config()["planning"])
    c.update(cell_px=5, robot_radius_px=10, safety_margin_px=10, validation_tolerance_px=5,
             validation_sample_step_px=1.0)
    c["goal_selection"] = {"pickup": {"max_dist_px": 40, "min_dist_px": 0, "distance_weight": 1.5, "top_k_report": 3},
                           "dropoff": {"max_dist_px": 40, "min_dist_px": 0, "distance_weight": 1.5, "top_k_report": 3}}
    return c


def rect(x1, y1, x2, y2):
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def simple_map(w=400, h=300, obstacles=(), no_go=(), start=(40, 150)):
    return {"schema_version": "stnav-map-0.1", "map_id": "test", "coordinate_frame": "raw_image_px",
            "image_size": [w, h], "review_status": "TEST", "traversable": [{"id": "T", "polygon": rect(10, 10, w - 10, h - 10)}],
            "static_obstacles": [{"id": f"O{i}", "polygon": p} for i, p in enumerate(obstacles)],
            "no_go": [{"id": f"N{i}", "polygon": p} for i, p in enumerate(no_go)],
            "start_pose": {"xy": list(start), "source": "test"}}
