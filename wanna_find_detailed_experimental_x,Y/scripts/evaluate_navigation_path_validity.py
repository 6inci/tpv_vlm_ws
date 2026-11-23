#!/usr/bin/env python3
"""Compare one preserved STEP 17 path against STEP 25N before/after worlds."""

from copy import deepcopy
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import sys


WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))

from vlm_reposition.planning import build_grid_map, load_planner_config, path_length_m, validate_current_path


STEP25N = WORKSPACE / "results" / "navigation_dynamic_world_integration"
PATH_SOURCE = WORKSPACE / "results" / "dynamic_path_invalidation" / "scenario_results" / "D1.json"
PLANNER_CONFIG = WORKSPACE / "config" / "planner_baseline.json"
STEP23M = WORKSPACE / "results" / "physical_measurement_registration" / "measurement_registration_summary.json"
OUTPUT = WORKSPACE / "results" / "navigation_path_validity"
PROTECTED_BASELINE = {
    "file_count": 478,
    "digest_sha256": "caf1084f799968b41db8ae62c82164dbc7e8f554c2505f29ef79dbf8ba70f466",
}


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def snapshot_protected():
    later_results = (
        WORKSPACE / "results" / "controlled_navigation_path_effect",
        WORKSPACE / "results" / "path_block_persistence_integration",
        WORKSPACE / "results" / "persistent_block_replanning_integration",
        WORKSPACE / "results" / "offline_navigation_chain_integration",
        WORKSPACE / "results" / "observed_blocking_acquisition",
    )
    files = sorted(
        path for path in (WORKSPACE / "results").rglob("*")
        if path.is_file() and OUTPUT not in path.parents
        and all(later not in path.parents for later in later_results)
    )
    files += [
        PLANNER_CONFIG,
        WORKSPACE / "config" / "dynamic_obstacle_scenarios.json",
        WORKSPACE / "config" / "measured_object_geometry_registry.json",
        WORKSPACE / "src" / "vlm_reposition" / "geometry" / "legacy.py",
        Path("/home/binci/qwen_vision_test/homography_12pt_undistorted.json"),
        Path("/home/binci/camera_calibration_1920x1080/calibration_result.json"),
    ]
    records = [(str(path), sha256_file(path)) for path in files]
    aggregate = hashlib.sha256("".join(f"{file_hash}  {path}\n" for path, file_hash in records).encode()).hexdigest()
    return {"file_count": len(records), "digest_sha256": aggregate}


def classify_path_effect(before_status, after_status):
    outcomes = {
        ("VALID", "BLOCKED"): "OBSERVATION_INDUCED_BLOCK",
        ("VALID", "VALID"): "NO_PATH_EFFECT",
        ("BLOCKED", "BLOCKED"): "PREEXISTING_BLOCK_OR_NONATTRIBUTABLE",
        ("BLOCKED", "VALID"): "UNEXPECTED_STATE_TRANSITION",
    }
    try:
        return outcomes[(before_status, after_status)]
    except KeyError as error:
        raise ValueError("path status must be VALID or BLOCKED") from error


def load_step25n_state():
    summary = read_json(STEP25N / "step25n_summary.json")
    before = read_json(STEP25N / "dynamic_world_before.json")
    after = read_json(STEP25N / "dynamic_world_after.json")
    obstacle = read_json(STEP25N / "dynamic_obstacle_record.json")
    effect = read_json(STEP25N / "occupancy_effect.json")
    if (
        summary.get("step_result") != "PASS"
        or summary.get("dynamic_obstacle") != "REGISTERED"
        or before.get("obstacle_state", {}).get("world_revision") != summary.get("world_revision_before")
        or after.get("obstacle_state", {}).get("world_revision") != summary.get("world_revision_after")
        or obstacle.get("id") != summary.get("dynamic_obstacle_id")
        or obstacle not in after["obstacle_state"]["obstacles"]
        or obstacle in before["obstacle_state"]["obstacles"]
    ):
        raise ValueError("STEP 25N artifacts are inconsistent")
    return summary, before, after, obstacle, effect


def load_existing_path():
    source = read_json(PATH_SOURCE)
    path = source.get("current_path")
    if (
        source.get("scenario_id") != "D1"
        or source.get("result") != "PASS"
        or source.get("automatic_replanning") != "NOT_RUN"
        or not source.get("current_path_unchanged_across_events")
        or not isinstance(path, dict)
        or path.get("status") != "PATH_FOUND"
        or not path.get("path_grid")
        or len(path["path_grid"]) != len(path.get("path_world", []))
    ):
        raise ValueError("canonical STEP 17 D1 current path is unavailable")
    for cell, point in zip(path["path_grid"], path["path_world"]):
        if (
            not isinstance(cell, list) or len(cell) != 2
            or not all(isinstance(value, int) and not isinstance(value, bool) for value in cell)
            or not isinstance(point, list) or len(point) != 2
            or any(isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) for value in point)
        ):
            raise ValueError("canonical current path contains invalid coordinates")
    path_digest = digest({"path_grid": path["path_grid"], "path_world": path["path_world"]})
    if path_digest != source.get("current_path_sha256"):
        raise ValueError("canonical current path digest mismatch")
    return source, path, path_digest


def blocking_provenance(before_result, after_result, path, occupancy_effect, dynamic_obstacle_id):
    if after_result["path_state"] != "BLOCKED":
        return {
            "status": "NOT_APPLICABLE_AFTER_PATH_VALID",
            "first_blocked_waypoint_index": None,
            "first_blocked_edge_index": None,
            "blocking_obstacle_ids": [],
            "dynamic_obstacle_attribution": "NOT_CLAIMED",
        }
    waypoint = after_result["first_blocked_waypoint_index"]
    cell = path["path_grid"][waypoint] if waypoint is not None else None
    changed_cells = {(item[0], item[1]) for item in occupancy_effect["changed_cells"] if item[2:] == [0, 1]}
    newly_occupied = cell is not None and tuple(cell) in changed_cells
    attributed = before_result["path_state"] == "VALID" and (
        dynamic_obstacle_id in after_result["blocking_obstacle_ids"] or newly_occupied
    )
    return {
        "status": "AVAILABLE",
        "first_blocked_waypoint_index": waypoint,
        "first_blocked_edge_index": after_result["first_blocked_edge_index"],
        "blocking_grid_cell": cell,
        "blocking_cell_newly_occupied": newly_occupied,
        "blocking_obstacle_ids": deepcopy(after_result["blocking_obstacle_ids"]),
        "dynamic_obstacle_attribution": "SUPPORTED" if attributed else "NOT_SUPPORTED",
    }


def main():
    protected_before = snapshot_protected()
    if protected_before != PROTECTED_BASELINE:
        raise RuntimeError(f"protected STEP 12-25N artifact baseline mismatch: {protected_before}")
    measurement = read_json(STEP23M)
    if measurement["measurement_input_status"] != "WAITING_FOR_USER_MEASUREMENT" or measurement["representative_physical_geometry"] != "NOT_READY":
        raise RuntimeError("STEP 23M physical branch changed")

    step25_summary, before_artifact, after_artifact, obstacle, occupancy = load_step25n_state()
    path_source, current_path, path_digest = load_existing_path()
    before_world, after_world = before_artifact["collision_world"], after_artifact["collision_world"]
    if before_world["coordinate_frame"] != after_world["coordinate_frame"]:
        raise RuntimeError("before/after world frames differ")
    planner = load_planner_config(PLANNER_CONFIG)
    before_grid = build_grid_map(before_world, planner)
    after_grid = build_grid_map(after_world, planner)
    if digest(before_grid["occupancy"]) != occupancy["occupancy_sha256_before"] or digest(after_grid["occupancy"]) != occupancy["occupancy_sha256_after"]:
        raise RuntimeError("STEP 25N occupancy reconstruction mismatch")

    before_path = {"path_grid": deepcopy(current_path["path_grid"]), "path_world": deepcopy(current_path["path_world"])}
    after_path = deepcopy(before_path)
    before_result = validate_current_path(before_path["path_grid"], before_path["path_world"], before_world, before_grid)
    after_result = validate_current_path(after_path["path_grid"], after_path["path_world"], after_world, after_grid)
    before_digest, after_digest = digest(before_path), digest(after_path)
    classification = classify_path_effect(before_result["path_state"], after_result["path_state"])
    provenance = blocking_provenance(before_result, after_result, current_path, occupancy, obstacle["id"])
    protected_after = snapshot_protected()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT / "path_source.json", {
        "source": str(PATH_SOURCE),
        "source_step": "STEP_17",
        "scenario_id": "D1",
        "planner": current_path["planner"],
        "coordinate_frame": before_world["coordinate_frame"],
        "path_sha256": path_digest,
        "waypoint_count": len(current_path["path_world"]),
        "path_length_m": current_path["path_length_m"],
        "recomputed_path_length_m": path_length_m(current_path["path_world"]),
        "new_path_generated": False,
    })
    write_json(OUTPUT / "step17_path_contract_audit.json", {
        "schema_version": "0.1",
        "path_representation": "ALIGNED_8_CONNECTED_GRID_CELLS_AND_CELL_CENTER_WORLD_POINTS",
        "validity_api": "vlm_reposition.planning.validate_current_path",
        "checks": ["waypoint workspace/collision/grid", "continuous segment/AABB intersection", "8-connected transition", "diagonal corner-cut"],
        "first_blocking_support": ["waypoint_index", "edge_index", "blocking_obstacle_ids"],
        "step18_input_dependency": "ACTIVE_PATH_PLUS_CURRENT_POSE_AND_UPDATED_STEP17_WORLD",
        "stronger_geometry_claimed": False,
    })
    write_json(OUTPUT / "path_before.json", {**before_path, "path_sha256": before_digest})
    write_json(OUTPUT / "path_after.json", {**after_path, "path_sha256": after_digest})
    write_json(OUTPUT / "path_validity_before.json", before_result)
    write_json(OUTPUT / "path_validity_after.json", after_result)
    write_json(OUTPUT / "blocking_provenance.json", provenance)
    write_json(OUTPUT / "path_effect_classification.json", {
        "before": before_result["path_state"],
        "after": after_result["path_state"],
        "classification": classification,
        "dynamic_obstacle_id": obstacle["id"],
        "attribution_status": provenance["dynamic_obstacle_attribution"],
    })
    summary = {
        "schema_version": "0.1",
        "step_result": "PASS",
        "software": "PASS",
        "path_source": "STEP17_D1_PRESERVED_CURRENT_PATH",
        "path_sha256_before": before_digest,
        "path_sha256_after": after_digest,
        "same_path_before_after": before_digest == after_digest == path_digest,
        "world_revision_before": before_result["world_revision"],
        "world_revision_after": after_result["world_revision"],
        "path_validity_before": before_result["path_state"],
        "path_validity_after": after_result["path_state"],
        "effect_classification": classification,
        "dynamic_obstacle_id": obstacle["id"],
        "blocking_provenance": provenance["status"],
        "source_cell_already_occupied_before": occupancy["source_cell_occupancy_before"] == 1,
        "step25n_changed_collision_cell_count": occupancy["changed_cell_count"],
        "physical_geometry": "NOT_READY_NOT_USED",
        "step23m_measurement_input_status": measurement["measurement_input_status"],
        "qwen_inference_runs": 0,
        "homography_recomputation": "NOT_RUN",
        "new_path_generation": "NOT_RUN",
        "step18_call": "NOT_RUN",
        "step19_call": "NOT_RUN",
        "astar": "NOT_RUN",
        "replanning": "NOT_RUN",
        "ros_nav2": "NOT_RUN",
        "protected_artifacts_before": protected_before,
        "protected_artifacts_after": protected_after,
        "protected_artifacts_unchanged": protected_before == protected_after,
        "recommended_next_step": "CONTROLLED_PATH_EFFECT_EXPERIMENT_USING_EXISTING_CONTRACTS",
    }
    if not summary["same_path_before_after"] or not summary["protected_artifacts_unchanged"]:
        raise RuntimeError("STEP 26N comparison integrity failed")
    write_json(OUTPUT / "step26n_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
