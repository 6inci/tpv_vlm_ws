#!/usr/bin/env python3
"""Compare the observed no-effect control with an existing STEP 17 blocker fixture."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys


WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))

from vlm_reposition.planning import (
    apply_obstacle_event,
    build_dynamic_collision_world,
    build_grid_map,
    dynamic_scenario_by_id,
    initial_obstacle_state,
    load_dynamic_config,
    load_planner_config,
    validate_current_path,
)
from vlm_reposition.world import load_physical_geometry, load_zones


STEP26N = WORKSPACE / "results" / "navigation_path_validity"
STEP17_RESULTS = WORKSPACE / "results" / "dynamic_path_invalidation" / "scenario_results"
DYNAMIC_CONFIG = WORKSPACE / "config" / "dynamic_obstacle_scenarios.json"
PLANNER_CONFIG = WORKSPACE / "config" / "planner_baseline.json"
ZONES = WORKSPACE / "config" / "zones.json"
PHYSICAL = WORKSPACE / "config" / "physical_geometry.json"
READINESS = WORKSPACE / "results" / "physical_geometry" / "physical_geometry_readiness.json"
STEP23M = WORKSPACE / "results" / "physical_measurement_registration" / "measurement_registration_summary.json"
OUTPUT = WORKSPACE / "results" / "controlled_navigation_path_effect"
PROTECTED_BASELINE = {
    "file_count": 487,
    "digest_sha256": "33c5753294ba2f16cd26f3b65a6fd4d65a71168f74335146361e57fa42cc4a32",
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
        DYNAMIC_CONFIG,
        PLANNER_CONFIG,
        WORKSPACE / "config" / "measured_object_geometry_registry.json",
        WORKSPACE / "src" / "vlm_reposition" / "geometry" / "legacy.py",
        Path("/home/binci/qwen_vision_test/homography_12pt_undistorted.json"),
        Path("/home/binci/camera_calibration_1920x1080/calibration_result.json"),
    ]
    records = [(str(path), sha256_file(path)) for path in files]
    aggregate = hashlib.sha256("".join(f"{file_hash}  {path}\n" for path, file_hash in records).encode()).hexdigest()
    return {"file_count": len(records), "digest_sha256": aggregate}


def directory_manifest(directory):
    records = [
        {"path": str(path), "sha256": sha256_file(path)}
        for path in sorted(Path(directory).rglob("*")) if path.is_file()
    ]
    return {
        "file_count": len(records),
        "digest_sha256": hashlib.sha256("".join(f'{item["sha256"]}  {item["path"]}\n' for item in records).encode()).hexdigest(),
        "files": records,
    }


def load_control_case():
    summary = read_json(STEP26N / "step26n_summary.json")
    before = read_json(STEP26N / "path_validity_before.json")
    after = read_json(STEP26N / "path_validity_after.json")
    effect = read_json(STEP26N / "path_effect_classification.json")
    source = read_json(STEP26N / "path_source.json")
    if (
        summary.get("step_result") != "PASS"
        or summary.get("path_validity_before") != "VALID"
        or summary.get("path_validity_after") != "VALID"
        or summary.get("effect_classification") != "NO_PATH_EFFECT"
        or before.get("path_state") != "VALID"
        or after.get("path_state") != "VALID"
        or effect.get("classification") != "NO_PATH_EFFECT"
        or source.get("path_sha256") != summary.get("path_sha256_before")
        or not summary.get("same_path_before_after")
    ):
        raise ValueError("STEP 26N observed control evidence is inconsistent")
    return {
        "case_id": "CASE_A_OBSERVED_CONTROL",
        "provenance_class": "OBSERVED_NAVIGATION_APPROXIMATION",
        "source_directory": str(STEP26N),
        "source_manifest": directory_manifest(STEP26N),
        "observation_id": "image_014_photo_08_track_039",
        "dynamic_obstacle_id": summary["dynamic_obstacle_id"],
        "path_sha256": source["path_sha256"],
        "before": before["path_state"],
        "after": after["path_state"],
        "effect": effect["classification"],
        "source_artifacts_modified": False,
    }


def audit_existing_fixtures():
    audited = []
    for path in sorted(STEP17_RESULTS.glob("D*.json")):
        result = read_json(path)
        for event in result["events"]:
            qualifies = (
                result.get("result") == "PASS"
                and result.get("automatic_replanning") == "NOT_RUN"
                and result.get("current_path_unchanged_across_events") is True
                and event.get("path_state_before") == "VALID"
                and event.get("path_state_after") == "BLOCKED"
            )
            audited.append({
                "scenario_id": result["scenario_id"],
                "event_id": event["event_id"],
                "artifact": str(path),
                "path_sha256": result["current_path_sha256"],
                "transition": f'{event["path_state_before"]}_TO_{event["path_state_after"]}',
                "qualifies_as_positive_fixture": qualifies,
            })
    candidates = [item for item in audited if item["qualifies_as_positive_fixture"]]
    if not candidates:
        raise RuntimeError("PARTIAL — NO_EXISTING_POSITIVE_FIXTURE")
    selected = next((item for item in candidates if item["scenario_id"] == "D2"), candidates[0])
    return audited, selected


def reproduce_positive_case(selected):
    stored = read_json(selected["artifact"])
    config = load_dynamic_config(DYNAMIC_CONFIG)
    planner = load_planner_config(PLANNER_CONFIG)
    zones = load_zones(ZONES)
    physical = load_physical_geometry(PHYSICAL)
    readiness = read_json(READINESS)["planning_geometry_readiness"]
    scenario = dynamic_scenario_by_id(config, selected["scenario_id"])
    event = scenario["events"][0]
    if event["event_id"] != selected["event_id"] or event["operation"] != "ADD":
        raise ValueError("selected STEP 17 positive fixture does not match its config")

    path_grid = deepcopy(stored["current_path"]["path_grid"])
    path_world = deepcopy(stored["current_path"]["path_world"])
    path_digest_before = digest({"path_grid": path_grid, "path_world": path_world})
    state_before = initial_obstacle_state(scenario)
    world_before = build_dynamic_collision_world(config, state_before, zones, physical, readiness)
    grid_before = build_grid_map(world_before, planner)
    validity_before = validate_current_path(path_grid, path_world, world_before, grid_before)
    state_after = apply_obstacle_event(state_before, event)
    world_after = build_dynamic_collision_world(config, state_after, zones, physical, readiness)
    grid_after = build_grid_map(world_after, planner)
    validity_after = validate_current_path(path_grid, path_world, world_after, grid_after)
    path_digest_after = digest({"path_grid": path_grid, "path_world": path_world})

    stored_before = stored["revisions"][0]["path_validation"]
    stored_after = stored["revisions"][1]["path_validation"]
    if (
        path_digest_before != stored["current_path_sha256"]
        or path_digest_after != path_digest_before
        or validity_before != stored_before
        or validity_after != stored_after
        or validity_before["path_state"] != "VALID"
        or validity_after["path_state"] != "BLOCKED"
    ):
        raise RuntimeError("documented positive fixture reproduction mismatch")

    waypoint = validity_after["first_blocked_waypoint_index"]
    cell = path_grid[waypoint] if waypoint is not None else None
    cell_before = grid_before["occupancy"][cell[0]][cell[1]] if cell else None
    cell_after = grid_after["occupancy"][cell[0]][cell[1]] if cell else None
    return {
        "stored": stored,
        "scenario": scenario,
        "event": event,
        "path_grid": path_grid,
        "path_world": path_world,
        "path_sha256": path_digest_before,
        "state_before": state_before,
        "state_after": state_after,
        "world_before": world_before,
        "world_after": world_after,
        "grid_before": grid_before,
        "grid_after": grid_after,
        "validity_before": validity_before,
        "validity_after": validity_after,
        "blocking_provenance": {
            "first_blocked_waypoint_index": waypoint,
            "first_blocked_edge_index": validity_after["first_blocked_edge_index"],
            "blocking_grid_cell": cell,
            "blocking_world_point_m": path_world[waypoint] if waypoint is not None else None,
            "blocking_cell_occupancy_before": cell_before,
            "blocking_cell_occupancy_after": cell_after,
            "blocking_cell_newly_occupied": cell_before == 0 and cell_after == 1,
            "blocking_obstacle_ids": deepcopy(validity_after["blocking_obstacle_ids"]),
        },
    }


def main():
    protected_before = snapshot_protected()
    if protected_before != PROTECTED_BASELINE:
        raise RuntimeError(f"protected STEP 12-26N artifact baseline mismatch: {protected_before}")
    measurement = read_json(STEP23M)
    if measurement["measurement_input_status"] != "WAITING_FOR_USER_MEASUREMENT" or measurement["representative_physical_geometry"] != "NOT_READY":
        raise RuntimeError("STEP 23M physical branch changed")

    control_manifest_before = directory_manifest(STEP26N)
    control = load_control_case()
    audited, selected = audit_existing_fixtures()
    positive = reproduce_positive_case(selected)
    control_manifest_after = directory_manifest(STEP26N)
    protected_after = snapshot_protected()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT / "fixture_audit.json", {"audited": audited, "selected": selected})
    write_json(OUTPUT / "control_case_reference.json", control)
    write_json(OUTPUT / "positive_fixture_selection.json", {
        "case_id": "CASE_B_DOCUMENTED_BLOCKING_FIXTURE",
        "provenance_class": "DOCUMENTED_DEVELOPMENT_FIXTURE",
        "scenario_id": selected["scenario_id"],
        "event_id": selected["event_id"],
        "source_artifact": selected["artifact"],
        "source_artifact_sha256": sha256_file(selected["artifact"]),
        "config_source": str(DYNAMIC_CONFIG),
        "config_source_sha256": sha256_file(DYNAMIC_CONFIG),
        "path_existed_before_step27n": True,
        "camera_or_vlm_derived": False,
        "fixture_modified": False,
    })
    write_json(OUTPUT / "positive_before_state.json", {
        "obstacle_state": positive["state_before"], "collision_world": positive["world_before"]
    })
    write_json(OUTPUT / "positive_after_state.json", {
        "obstacle_state": positive["state_after"], "collision_world": positive["world_after"]
    })
    write_json(OUTPUT / "positive_path_validity_before.json", positive["validity_before"])
    write_json(OUTPUT / "positive_path_validity_after.json", positive["validity_after"])
    write_json(OUTPUT / "positive_blocking_provenance.json", positive["blocking_provenance"])
    comparison = {
        "control_case": {
            "provenance": control["provenance_class"],
            "path_sha256": control["path_sha256"],
            "before": control["before"], "after": control["after"], "effect": control["effect"],
        },
        "positive_case": {
            "provenance": "DOCUMENTED_DEVELOPMENT_FIXTURE",
            "path_sha256_before": positive["path_sha256"],
            "path_sha256_after": positive["path_sha256"],
            "before": positive["validity_before"]["path_state"],
            "after": positive["validity_after"]["path_state"],
            "effect": "PATH_BLOCKED",
            "blocking_obstacle_ids": positive["validity_after"]["blocking_obstacle_ids"],
        },
        "interpretation": {
            "obstacle_presence_implies_block": False,
            "blocking_detection_contract_verified": True,
            "evidence_classes_combined": False,
        },
    }
    write_json(OUTPUT / "controlled_effect_comparison.json", comparison)
    summary = {
        "schema_version": "0.1",
        "step_result": "PASS",
        "software": "PASS",
        "control_case": "VALID_TO_VALID_NO_PATH_EFFECT",
        "positive_case": "VALID_TO_BLOCKED_PATH_BLOCKED",
        "positive_fixture": f'{selected["scenario_id"]}_{selected["event_id"]}',
        "positive_path_same_before_after": positive["path_sha256"] == selected["path_sha256"],
        "positive_blocking_obstacle_ids": positive["validity_after"]["blocking_obstacle_ids"],
        "observed_and_fixture_provenance_separate": True,
        "control_artifacts_before": control_manifest_before,
        "control_artifacts_after": control_manifest_after,
        "control_artifacts_unchanged": control_manifest_before == control_manifest_after,
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
        "recommended_next_step": "BLOCKED_CONDITION_TO_EXISTING_PERSISTENCE_STOP_WAIT_POLICY_INTEGRATION",
    }
    if not summary["control_artifacts_unchanged"] or not summary["protected_artifacts_unchanged"]:
        raise RuntimeError("STEP 27N evidence protection failed")
    write_json(OUTPUT / "step27n_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
