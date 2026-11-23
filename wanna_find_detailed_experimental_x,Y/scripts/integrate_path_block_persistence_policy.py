#!/usr/bin/env python3
"""Connect preserved STEP 27N path effects to STEP 19 without replanning."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys


WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))

from vlm_reposition.planning import (
    build_grid_map,
    dynamic_scenario_by_id,
    evaluate_policy_observation,
    load_dynamic_config,
    load_obstacle_policy_config,
    load_planner_config,
    policy_scenario_by_id,
)
from vlm_reposition.world import load_physical_geometry, load_zones


STEP27N = WORKSPACE / "results" / "controlled_navigation_path_effect"
STEP25N = WORKSPACE / "results" / "navigation_dynamic_world_integration"
STEP17_RESULTS = WORKSPACE / "results" / "dynamic_path_invalidation" / "scenario_results"
STEP19_RESULTS = WORKSPACE / "results" / "dynamic_obstacle_policy"
STEP23M = WORKSPACE / "results" / "physical_measurement_registration" / "measurement_registration_summary.json"
DYNAMIC_CONFIG = WORKSPACE / "config" / "dynamic_obstacle_scenarios.json"
POLICY_CONFIG = WORKSPACE / "config" / "dynamic_obstacle_policy.json"
PLANNER_CONFIG = WORKSPACE / "config" / "planner_baseline.json"
ZONES = WORKSPACE / "config" / "zones.json"
PHYSICAL = WORKSPACE / "config" / "physical_geometry.json"
READINESS = WORKSPACE / "results" / "physical_geometry" / "physical_geometry_readiness.json"
OUTPUT = WORKSPACE / "results" / "path_block_persistence_integration"
PROTECTED_BASELINE = {
    "file_count": 500,
    "digest_sha256": "1999dad56cffda82106cc3409f1905e197aee2cd2699c622da29bc7e40af9628",
}


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def directory_manifest(directory):
    records = [
        {"path": str(path), "sha256": sha256_file(path)}
        for path in sorted(Path(directory).rglob("*")) if path.is_file()
    ]
    return {
        "file_count": len(records),
        "digest_sha256": hashlib.sha256(
            "".join(f'{item["sha256"]}  {item["path"]}\n' for item in records).encode()
        ).hexdigest(),
        "files": records,
    }


def snapshot_protected():
    later_results = (
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
        POLICY_CONFIG,
        PLANNER_CONFIG,
        WORKSPACE / "config" / "measured_object_geometry_registry.json",
        WORKSPACE / "src" / "vlm_reposition" / "planning" / "obstacle_policy.py",
        WORKSPACE / "src" / "vlm_reposition" / "planning" / "replanning.py",
        WORKSPACE / "src" / "vlm_reposition" / "geometry" / "legacy.py",
        Path("/home/binci/qwen_vision_test/homography_12pt_undistorted.json"),
        Path("/home/binci/camera_calibration_1920x1080/calibration_result.json"),
    ]
    records = [(str(path), sha256_file(path)) for path in files]
    aggregate = hashlib.sha256(
        "".join(f"{file_hash}  {path}\n" for path, file_hash in records).encode()
    ).hexdigest()
    return {"file_count": len(records), "digest_sha256": aggregate}


def load_step27_evidence():
    summary = read_json(STEP27N / "step27n_summary.json")
    comparison = read_json(STEP27N / "controlled_effect_comparison.json")
    control = read_json(STEP27N / "control_case_reference.json")
    selection = read_json(STEP27N / "positive_fixture_selection.json")
    blocking = read_json(STEP27N / "positive_blocking_provenance.json")
    if (
        summary.get("step_result") != "PASS"
        or comparison["control_case"] != {
            "provenance": "OBSERVED_NAVIGATION_APPROXIMATION",
            "path_sha256": control["path_sha256"],
            "before": "VALID", "after": "VALID", "effect": "NO_PATH_EFFECT",
        }
        or comparison["positive_case"]["provenance"] != "DOCUMENTED_DEVELOPMENT_FIXTURE"
        or comparison["positive_case"]["before"] != "VALID"
        or comparison["positive_case"]["after"] != "BLOCKED"
        or comparison["positive_case"]["blocking_obstacle_ids"] != ["dynamic_box_02"]
        or selection["scenario_id"] != "D2"
        or selection["event_id"] != "D2_add_on_path"
        or blocking["first_blocked_waypoint_index"] != 50
        or blocking["first_blocked_edge_index"] != 49
    ):
        raise RuntimeError("STEP 27N control/positive evidence mismatch")
    return summary, comparison, control, selection, blocking


def audit_step19_contract(policy):
    p1 = read_json(STEP19_RESULTS / "scenario_results" / "P1_temporary_clear.json")
    p2 = read_json(STEP19_RESULTS / "scenario_results" / "P2_persistent_replan.json")
    source = WORKSPACE / "src" / "vlm_reposition" / "planning" / "obstacle_policy.py"
    return {
        "source_of_truth": {
            "config": str(POLICY_CONFIG),
            "config_sha256": sha256_file(POLICY_CONFIG),
            "implementation": str(source),
            "implementation_sha256": sha256_file(source),
        },
        "input_schema": [
            "observation_index", "logical_time_s", "robot_waypoint_index", "detected_obstacle_ids"
        ],
        "identity_key": "obstacle_id",
        "parameters": deepcopy(policy["parameters"]),
        "logical_time_convention": "observation_index * observation_interval_s",
        "navigation_states": sorted({
            item["expected_navigation_state"]
            for scenario in policy["scenarios"] for item in scenario["observations"]
        }),
        "actions": sorted({
            item["expected_action"]
            for scenario in policy["scenarios"] for item in scenario["observations"]
        }),
        "presence_confirmation_fixture": {
            "scenario_id": "P1", "confirmation_count": 2,
            "confirmation_time_s": p1["obstacle_trackers"]["p1_box"]["presence_confirmation_history"][0]["confirmation_time_s"],
        },
        "absence_resume_fixture": {
            "scenario_id": "P1", "confirmation_count": 2,
            "final_action": p1["observations"][-1]["action"],
            "confirmation_time_s": p1["obstacle_trackers"]["p1_box"]["clear_confirmation_history"][0]["confirmation_time_s"],
        },
        "persistence_fixture": {
            "scenario_id": "P2",
            "logical_times_s": [item["logical_time_s"] for item in p2["observations"]],
            "classification": p2["blocking_episodes"][0]["classification"],
            "threshold_reached_at_s": p2["blocking_episodes"][0]["persistent_classification_time_s"],
        },
        "step18_handoff": "evaluate_event_and_replan after REPLAN_REQUIRED",
        "step28n_handoff_mode": "DECISION_ONLY_EXECUTE_REPLANNING_FALSE",
    }


def active_path_from_step17(scenario_id):
    stored = read_json(STEP17_RESULTS / f"{scenario_id}.json")
    path = stored["current_path"]
    path_hash = digest({"path_grid": path["path_grid"], "path_world": path["path_world"]})
    if path_hash != stored["current_path_sha256"]:
        raise RuntimeError(f"{scenario_id} current path digest mismatch")
    active = {
        "path_id": f"{scenario_id}_preserved_current_path",
        "path_revision": 0,
        "source_planner": path["planner"],
        "world_revision_created": 0,
        "path_grid": deepcopy(path["path_grid"]),
        "path_world": deepcopy(path["path_world"]),
        "path_length_m": path["path_length_m"],
        "minimum_clearance_m": path["minimum_clearance_m"],
        "accumulated_safety_risk": path["accumulated_safety_risk"],
        "waypoint_count": path["number_of_waypoints"],
        "status": "ACTIVE",
        "lineage": None,
    }
    return active, path_hash


def policy_runtime(state, world, grid, active):
    return {
        "obstacle_state": deepcopy(state),
        "world": deepcopy(world),
        "grid": deepcopy(grid),
        "active_path": deepcopy(active),
        "initial_active_path": deepcopy(active),
        "path_history": [],
        "trackers": {},
        "episodes": [],
        "active_episode_index": None,
        "navigation_state": "MOVING",
        "stopped_pose": None,
        "progress_by_revision": {},
        "planner_run_count": 0,
        "stop_count": 0,
        "wait_observation_count": 0,
        "initial_path_validation": None,
    }


def run_control(policy, planner, zones, physical, readiness, control):
    source = read_json(STEP25N / "dynamic_world_after.json")
    active, path_hash = active_path_from_step17("D1")
    if path_hash != control["path_sha256"]:
        raise RuntimeError("CASE A path identity mismatch")
    grid = build_grid_map(source["collision_world"], planner)
    runtime = policy_runtime(source["obstacle_state"], source["collision_world"], grid, active)
    scenario = {"scenario_id": "STEP28_CASE_A", "obstacle_catalog": []}
    observation = {
        "observation_index": 0, "logical_time_s": 0.0,
        "robot_waypoint_index": 0, "detected_obstacle_ids": [],
    }
    runtime, result = evaluate_policy_observation(
        policy, scenario, planner, zones, physical, readiness, runtime, observation,
        execute_replanning=False,
    )
    if (
        result["path_state"] != "VALID"
        or result["navigation_state"] != "MOVING"
        or result["action"] != "CONTINUE_CURRENT_PATH"
        or runtime["planner_run_count"] != 0
    ):
        raise RuntimeError("CASE A generated a false blocking policy trigger")
    policy_input = {
        "case_id": "CASE_A_OBSERVED_CONTROL",
        "evidence_class": control["provenance_class"],
        "observation_id": control["observation_id"],
        "obstacle_id": control["dynamic_obstacle_id"],
        "path_sha256": path_hash,
        "path_validity": "VALID",
        "effect_classification": "NO_PATH_EFFECT",
        "path_blocked": False,
        "world_revision": source["obstacle_state"]["world_revision"],
        "source": str(STEP27N / "control_case_reference.json"),
    }
    return policy_input, result


def run_blocked(policy, planner, zones, physical, readiness, comparison, selection, blocking):
    dynamic = load_dynamic_config(DYNAMIC_CONFIG)
    d2 = dynamic_scenario_by_id(dynamic, "D2")
    event = d2["events"][0]
    before = read_json(STEP27N / "positive_before_state.json")
    active, path_hash = active_path_from_step17("D2")
    if path_hash != comparison["positive_case"]["path_sha256_before"]:
        raise RuntimeError("CASE B path identity mismatch")
    grid = build_grid_map(before["collision_world"], planner)
    runtime = policy_runtime(before["obstacle_state"], before["collision_world"], grid, active)
    scenario = {
        "scenario_id": "STEP28_D2_POLICY_BRIDGE",
        "goal": deepcopy(d2["initial_goal"]),
        "obstacle_catalog": [deepcopy(event["obstacle"])],
    }
    p2 = policy_scenario_by_id(policy, "P2")
    observations = [
        {
            "observation_index": item["observation_index"],
            "logical_time_s": item["logical_time_s"],
            "robot_waypoint_index": item["robot_waypoint_index"],
            "detected_obstacle_ids": ["dynamic_box_02"] if item["detected_obstacle_ids"] else [],
        }
        for item in p2["observations"]
    ]
    trace = []
    for observation in observations:
        runtime, result = evaluate_policy_observation(
            policy, scenario, planner, zones, physical, readiness, runtime, observation,
            execute_replanning=False,
        )
        trace.append(result)
    final = trace[-1]
    if (
        trace[0]["path_state"] != "VALID"
        or trace[1]["path_state"] != "BLOCKED"
        or trace[1]["navigation_state"] != "STOPPED_FOR_OBSTACLE"
        or not all(item["action"] == "STOP_AND_WAIT" for item in trace[1:])
        or final["classification"] != "PERSISTENT_BLOCKING"
        or final["replan_status"] != "REPLAN_REQUIRED"
        or runtime["planner_run_count"] != 0
        or runtime["active_path"]["path_revision"] != 0
    ):
        raise RuntimeError("CASE B did not follow the existing decision-only STEP 19 policy")
    policy_input = {
        "case_id": "CASE_B_DOCUMENTED_BLOCKING_FIXTURE",
        "evidence_class": selection["provenance_class"],
        "scenario_id": selection["scenario_id"],
        "event_id": selection["event_id"],
        "obstacle_id": "dynamic_box_02",
        "path_sha256": path_hash,
        "path_validity": "BLOCKED",
        "effect_classification": "PATH_BLOCKED",
        "path_blocked": True,
        "world_revision": 1,
        "blocking_provenance": deepcopy(blocking),
        "blocking_provenance_index_frame": "STEP27_FULL_PATH_GLOBAL_INDEX",
        "logical_sequence_source": "STEP19_P2_DOCUMENTED_DEVELOPMENT_SEQUENCE",
        "source": selection["source_artifact"],
        "camera_or_vlm_temporal_evidence": False,
    }
    return policy_input, observations, trace, runtime


def main():
    protected_before = snapshot_protected()
    if protected_before != PROTECTED_BASELINE:
        raise RuntimeError(f"protected STEP 12-27N artifact baseline mismatch: {protected_before}")
    measurement = read_json(STEP23M)
    if measurement["measurement_input_status"] != "WAITING_FOR_USER_MEASUREMENT":
        raise RuntimeError("STEP 23M measurement state changed")
    step27_before = directory_manifest(STEP27N)
    _, comparison, control, selection, blocking = load_step27_evidence()
    policy = load_obstacle_policy_config(POLICY_CONFIG)
    planner = load_planner_config(PLANNER_CONFIG)
    zones = load_zones(ZONES)
    physical = load_physical_geometry(PHYSICAL)
    readiness = read_json(READINESS)["planning_geometry_readiness"]
    contract = audit_step19_contract(policy)
    control_input, control_result = run_control(policy, planner, zones, physical, readiness, control)
    blocked_input, sequence, trace, runtime = run_blocked(
        policy, planner, zones, physical, readiness, comparison, selection, blocking
    )
    step27_after = directory_manifest(STEP27N)
    protected_after = snapshot_protected()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT / "step19_contract_audit.json", contract)
    write_json(OUTPUT / "control_policy_input.json", control_input)
    write_json(OUTPUT / "control_policy_result.json", control_result)
    write_json(OUTPUT / "blocked_policy_input.json", blocked_input)
    write_json(OUTPUT / "policy_sequence.json", {
        "persistent_sequence_source": "STEP19_P2",
        "identity_remap": {"p2_box": "dynamic_box_02"},
        "logical_time_evidence": "DOCUMENTED_DEVELOPMENT_POLICY_FIXTURE",
        "observations": sequence,
        "absence_resume_reference": {
            "source": str(STEP19_RESULTS / "scenario_results" / "P1_temporary_clear.json"),
            "evidence_class": "DEVELOPMENT_POLICY_FIXTURE",
            "confirmation_count": policy["parameters"]["absence_confirmation_count"],
            "documented_action": "RESUME_CURRENT_PATH",
            "case_b_disappearance_executed": False,
        },
    })
    write_json(OUTPUT / "policy_transition_trace.json", {
        "source_blocking_provenance": deepcopy(blocking),
        "step19_remaining_path_index_offset": sequence[0]["robot_waypoint_index"],
        "revalidated_blocking_index_frame": "REMAINING_PATH_RELATIVE_INDEX",
        "observations": trace,
    })
    episode = runtime["episodes"][0]
    write_json(OUTPUT / "persistence_decision.json", {
        "classification": episode["classification"],
        "blocking_since_s": episode["blocking_since_s"],
        "blocking_duration_s": episode["blocking_duration_s"],
        "persistent_classification_time_s": episode["persistent_classification_time_s"],
        "replan_status": episode["replan_status"],
        "decision_source": "EXISTING_STEP19_POLICY",
        "single_block_sample_manually_promoted": False,
        "live_temporal_evidence": False,
    })
    write_json(OUTPUT / "replan_decision_boundary.json", {
        "decision": "REPLAN_REQUIRED",
        "step18_handoff": "NOT_INVOKED",
        "replanning_execution": "NOT_RUN",
        "planner_runs": runtime["planner_run_count"],
        "new_path_generation": "NOT_RUN",
        "active_path_revision": runtime["active_path"]["path_revision"],
        "hardware_stop": "NOT_CLAIMED",
    })
    summary = {
        "schema_version": "0.1",
        "step_result": "PASS",
        "control_policy": "VALID_CONTINUE_CURRENT_PATH",
        "blocked_policy": "STOP_WAIT_PERSISTENT_REPLAN_REQUIRED",
        "case_b_final_navigation_state": trace[-1]["navigation_state"],
        "case_b_final_action": trace[-1]["action"],
        "case_b_replan_decision": trace[-1]["replan_status"],
        "presence_confirmation_count": policy["parameters"]["presence_confirmation_count"],
        "absence_confirmation_count": policy["parameters"]["absence_confirmation_count"],
        "persistence_threshold_s": policy["parameters"]["persistence_threshold_s"],
        "observation_interval_s": policy["parameters"]["observation_interval_s"],
        "planner_runs": runtime["planner_run_count"],
        "step18_call": "NOT_RUN",
        "astar": "NOT_RUN",
        "new_path_generation": "NOT_RUN",
        "qwen_inference_runs": 0,
        "homography_recomputation": "NOT_RUN",
        "ros_nav2": "NOT_RUN",
        "hardware_stop_verified": False,
        "physical_geometry": "NOT_READY_NOT_USED",
        "step23m_measurement_input_status": measurement["measurement_input_status"],
        "observed_and_development_evidence_separate": True,
        "step27_artifacts_before": step27_before,
        "step27_artifacts_after": step27_after,
        "step27_artifacts_unchanged": step27_before == step27_after,
        "protected_artifacts_before": protected_before,
        "protected_artifacts_after": protected_after,
        "protected_artifacts_unchanged": protected_before == protected_after,
        "recommended_next_step": "PERSISTENT_BLOCK_REPLAN_REQUIRED_TO_EXISTING_STEP18_INTEGRATION",
    }
    if not summary["step27_artifacts_unchanged"] or not summary["protected_artifacts_unchanged"]:
        raise RuntimeError("STEP 28N evidence protection failed")
    write_json(OUTPUT / "step28n_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
