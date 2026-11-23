#!/usr/bin/env python3
"""Gate STEP 28N's persistent block decision into the existing STEP 18 API."""

from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path
import sys


WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))
sys.path.insert(0, str(WORKSPACE / "scripts"))

from integrate_path_block_persistence_policy import active_path_from_step17, directory_manifest
from vlm_reposition.planning import (
    dynamic_scenario_by_id,
    evaluate_event_and_replan,
    load_dynamic_config,
    load_obstacle_policy_config,
    load_planner_config,
)
from vlm_reposition.world import load_physical_geometry, load_zones


STEP28N = WORKSPACE / "results" / "path_block_persistence_integration"
STEP27N = WORKSPACE / "results" / "controlled_navigation_path_effect"
STEP18_RESULTS = WORKSPACE / "results" / "automatic_replanning" / "scenario_results"
STEP23M = WORKSPACE / "results" / "physical_measurement_registration" / "measurement_registration_summary.json"
DYNAMIC_CONFIG = WORKSPACE / "config" / "dynamic_obstacle_scenarios.json"
POLICY_CONFIG = WORKSPACE / "config" / "dynamic_obstacle_policy.json"
PLANNER_CONFIG = WORKSPACE / "config" / "planner_baseline.json"
ZONES = WORKSPACE / "config" / "zones.json"
PHYSICAL = WORKSPACE / "config" / "physical_geometry.json"
READINESS = WORKSPACE / "results" / "physical_geometry" / "physical_geometry_readiness.json"
OUTPUT = WORKSPACE / "results" / "persistent_block_replanning_integration"
STEP18_SOURCE = WORKSPACE / "src" / "vlm_reposition" / "planning" / "replanning.py"


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
        WORKSPACE / "results" / "offline_navigation_chain_integration",
        WORKSPACE / "results" / "observed_blocking_acquisition",
    )
    records = [
        (str(path), sha256_file(path))
        for path in sorted((WORKSPACE / "results").rglob("*"))
        if path.is_file() and OUTPUT not in path.parents
        and all(later not in path.parents for later in later_results)
    ]
    return {
        "file_count": len(records),
        "digest_sha256": hashlib.sha256(
            "".join(f"{file_hash}  {path}\n" for path, file_hash in records).encode()
        ).hexdigest(),
    }


def evaluate_replan_gate(policy_input, persistence, boundary):
    reasons = []
    required = {
        "path_validity": (policy_input or {}).get("path_validity"),
        "effect_classification": (policy_input or {}).get("effect_classification"),
        "persistence_classification": (persistence or {}).get("classification"),
        "decision": (boundary or {}).get("decision"),
    }
    expected = {
        "path_validity": "BLOCKED",
        "effect_classification": "PATH_BLOCKED",
        "persistence_classification": "PERSISTENT_BLOCKING",
        "decision": "REPLAN_REQUIRED",
    }
    reasons.extend(f"{key}_REJECTED" for key, value in required.items() if value != expected[key])
    provenance = (policy_input or {}).get("blocking_provenance")
    if (
        not isinstance(provenance, dict)
        or provenance.get("blocking_obstacle_ids") != ["dynamic_box_02"]
        or not isinstance(provenance.get("first_blocked_waypoint_index"), int)
        or not isinstance(provenance.get("first_blocked_edge_index"), int)
    ):
        reasons.append("MALFORMED_BLOCKING_PROVENANCE")
    if (policy_input or {}).get("evidence_class") != "DOCUMENTED_DEVELOPMENT_FIXTURE":
        reasons.append("EVIDENCE_CLASS_REJECTED")
    if (persistence or {}).get("decision_source") != "EXISTING_STEP19_POLICY":
        reasons.append("DECISION_PROVENANCE_REJECTED")
    return {
        "eligible": not reasons,
        "required_values": expected,
        "observed_values": required,
        "rejection_reasons": reasons,
    }


def audit_step18():
    failed = read_json(STEP18_RESULTS / "R4_no_path_stop.json")["events"][0]
    return {
        "implementation": str(STEP18_SOURCE),
        "implementation_sha256": sha256_file(STEP18_SOURCE),
        "entry_api": "evaluate_event_and_replan",
        "signature": str(inspect.signature(evaluate_event_and_replan)),
        "world_state_input": "obstacle_state with world_revision and obstacles",
        "current_path_input": "active_path path_grid/path_world and path_revision",
        "robot_progress": "event.robot_waypoint_index on the active path revision",
        "goal": "scenario.goal in LEGACY_WORKSPACE_COORDINATE meters",
        "trigger": "event with event_id/event_index/operation/robot_waypoint_index",
        "world_update_mode": "apply_obstacle_update=False because STEP28N world revision 1 is already confirmed",
        "planner": "existing SAFETY_COST_ASTAR via plan_path",
        "acceptance": "existing validate_replanned_path must return VALID",
        "success_revision": "old path SUPERSEDED; accepted active path revision = old revision + 1",
        "failure_contract": {
            "historical_source": str(STEP18_RESULTS / "R4_no_path_stop.json"),
            "replan_status": failed["replanning"]["status"],
            "system_action": failed["system_action"],
            "stop_required": failed["stop_required"],
            "new_active_path": failed["new_active_path"],
        },
        "dependencies": {
            "step17": "dynamic world and current-path validity",
            "step19": "persistent REPLAN_REQUIRED decision",
        },
    }


def main():
    protected_before = snapshot_protected()
    step28_before = directory_manifest(STEP28N)
    planner_hash_before = sha256_file(PLANNER_CONFIG)
    measurement = read_json(STEP23M)
    if measurement["measurement_input_status"] != "WAITING_FOR_USER_MEASUREMENT":
        raise RuntimeError("STEP 23M measurement state changed")

    control_input = read_json(STEP28N / "control_policy_input.json")
    control_result = read_json(STEP28N / "control_policy_result.json")
    blocked_input = read_json(STEP28N / "blocked_policy_input.json")
    persistence = read_json(STEP28N / "persistence_decision.json")
    boundary = read_json(STEP28N / "replan_decision_boundary.json")
    trace = read_json(STEP28N / "policy_transition_trace.json")["observations"]
    control_gate = evaluate_replan_gate(control_input, None, None)
    blocked_gate = evaluate_replan_gate(blocked_input, persistence, boundary)
    if control_gate["eligible"] or not blocked_gate["eligible"]:
        raise RuntimeError("STEP 29N replanning gate rejected the required control/positive split")
    if control_result["action"] != "CONTINUE_CURRENT_PATH" or control_result["active_path_revision"] != 0:
        raise RuntimeError("CASE A control state changed")

    dynamic = load_dynamic_config(DYNAMIC_CONFIG)
    d2 = dynamic_scenario_by_id(dynamic, "D2")
    policy = load_obstacle_policy_config(POLICY_CONFIG)
    planner = load_planner_config(PLANNER_CONFIG)
    zones = load_zones(ZONES)
    physical = load_physical_geometry(PHYSICAL)
    readiness = read_json(READINESS)["planning_geometry_readiness"]
    after = read_json(STEP27N / "positive_after_state.json")
    active, path_hash = active_path_from_step17("D2")
    final_policy = trace[-1]
    waypoint = final_policy["current_robot_pose"]["waypoint_index"]
    if (
        path_hash != blocked_input["path_sha256"]
        or after["obstacle_state"]["world_revision"] != blocked_input["world_revision"]
        or active["path_world"][waypoint] != [final_policy["current_robot_pose"]["x"], final_policy["current_robot_pose"]["y"]]
    ):
        raise RuntimeError("CASE B established path/world/progress identity mismatch")

    scenario = {"scenario_id": "D2", "goal": deepcopy(d2["initial_goal"])}
    trigger = {
        "event_id": f"D2_persistence_{final_policy['blocking_episode_id']}",
        "event_index": final_policy["observation_index"],
        "operation": "PERSISTENCE_TRIGGER",
        "robot_waypoint_index": waypoint,
    }
    old_path = deepcopy(active)
    state, world, grid, replacement, history, result = evaluate_event_and_replan(
        policy, scenario, planner, zones, physical, readiness,
        after["obstacle_state"], active, [], trigger, waypoint,
        apply_obstacle_update=False,
    )
    invocation_count = 1
    if result["planner_runs_after_event"] != 1 or result["old_path_state"] != "BLOCKED":
        raise RuntimeError("existing STEP18 did not process the persistent blocked path")
    outcome = result["replanning"]["status"]
    if outcome == "REPLAN_SUCCESS":
        if replacement is None or result["replanned_path_validation"]["status"] != "VALID":
            raise RuntimeError("STEP18 success did not produce an accepted replacement path")
    elif outcome == "REPLAN_FAILED":
        if replacement is not None or result["system_action"] != "STOP":
            raise RuntimeError("STEP18 failure contract changed")
    else:
        raise RuntimeError(f"unexpected STEP18 outcome: {outcome}")

    step28_after = directory_manifest(STEP28N)
    protected_after = snapshot_protected()
    planner_hash_after = sha256_file(PLANNER_CONFIG)
    if step28_before != step28_after or protected_before != protected_after or planner_hash_before != planner_hash_after:
        raise RuntimeError("protected evidence or planner configuration changed")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT / "step18_contract_audit.json", audit_step18())
    write_json(OUTPUT / "control_replan_gate.json", {
        "case_id": control_input["case_id"], "evidence_class": control_input["evidence_class"],
        "path_validity": control_input["path_validity"], "policy_action": control_result["action"],
        "gate": control_gate, "step18_invocations": 0, "planner_runs": 0,
        "new_path": "NOT_GENERATED", "path_revision_before": 0, "path_revision_after": 0,
    })
    write_json(OUTPUT / "blocked_replan_input.json", {
        "case_id": blocked_input["case_id"], "evidence_class": blocked_input["evidence_class"],
        "path_validity": blocked_input["path_validity"], "persistence": persistence["classification"],
        "decision": boundary["decision"], "decision_source": persistence["decision_source"],
        "obstacle_id": blocked_input["obstacle_id"], "world_revision": blocked_input["world_revision"],
        "path_sha256": path_hash, "blocking_provenance": deepcopy(blocked_input["blocking_provenance"]),
        "camera_or_vlm_temporal_evidence": False, "gate": blocked_gate,
    })
    write_json(OUTPUT / "step18_handoff.json", {
        "implementation": "EXISTING_STEP18", "api": "evaluate_event_and_replan",
        "invocation_count": invocation_count, "apply_obstacle_update": False,
        "source_active_path_id": old_path["path_id"], "source_path_revision": old_path["path_revision"],
        "current_robot_progress_source": str(STEP28N / "policy_transition_trace.json"),
        "current_robot_pose": deepcopy(final_policy["current_robot_pose"]),
        "goal": deepcopy(scenario["goal"]), "world_revision": state["world_revision"],
        "trigger": trigger, "planner_config": str(PLANNER_CONFIG),
        "planner_config_sha256": planner_hash_before,
    })
    write_json(OUTPUT / "replan_execution_result.json", {
        "status": outcome, "step18_result": result,
        "world_revision": state["world_revision"], "planner_runs": result["planner_runs_after_event"],
        "system_action": result["system_action"], "stop_required": result["stop_required"],
    })
    write_json(OUTPUT / "old_path_reference.json", {
        "source": str(WORKSPACE / "results" / "dynamic_path_invalidation" / "scenario_results" / "D2.json"),
        "path_sha256": path_hash, "path": old_path,
        "history_status_after_step18": history[0]["status"],
    })
    write_json(OUTPUT / "replacement_path.json", {
        "status": "GENERATED" if replacement is not None else "NOT_GENERATED",
        "path": replacement,
    })
    write_json(OUTPUT / "replacement_path_validation.json", {
        "status": "VALID" if replacement is not None else "NOT_APPLICABLE_REPLAN_FAILED",
        "existing_step18_validation": result["replanned_path_validation"],
        "coordinate_frame": policy["coordinate_frame"], "unit": policy["unit"],
        "dynamic_world_revision": state["world_revision"],
    })
    write_json(OUTPUT / "path_revision_result.json", {
        "old_path_id": old_path["path_id"], "old_revision": old_path["path_revision"],
        "old_status_after_step18": history[0]["status"],
        "new_path_id": None if replacement is None else replacement["path_id"],
        "new_revision": None if replacement is None else replacement["path_revision"],
        "existing_step18_semantics_followed": True,
    })
    summary = {
        "schema_version": "0.1", "step_result": "PASS",
        "case_a_step18_invocations": 0, "case_a_planner_runs": 0, "case_a_path_revision": 0,
        "case_b_evidence_class": blocked_input["evidence_class"],
        "case_b_step18_invocations": invocation_count, "case_b_planner_runs": result["planner_runs_after_event"],
        "case_b_replanning_result": outcome, "case_b_system_action": result["system_action"],
        "old_path_waypoint_count": old_path["waypoint_count"], "old_path_length_m": old_path["path_length_m"],
        "new_path_waypoint_count": None if replacement is None else replacement["waypoint_count"],
        "new_path_length_m": None if replacement is None else replacement["path_length_m"],
        "old_path_revision": old_path["path_revision"],
        "new_path_revision": None if replacement is None else replacement["path_revision"],
        "qwen_inference_runs": 0, "homography": "NOT_RUN", "ros_nav2": "NOT_RUN",
        "hardware_execution": "NOT_RUN", "physical_geometry": "NOT_READY_NOT_INFERRED",
        "step23m_measurement_input_status": measurement["measurement_input_status"],
        "planner_config_unchanged": planner_hash_before == planner_hash_after,
        "step28_artifacts_before": step28_before, "step28_artifacts_after": step28_after,
        "step28_artifacts_unchanged": step28_before == step28_after,
        "protected_artifacts_before": protected_before, "protected_artifacts_after": protected_after,
        "protected_artifacts_unchanged": protected_before == protected_after,
        "verification": {},
        "recommended_next_step": "OFFLINE_END_TO_END_NAVIGATION_INTEGRATION_WITH_EVIDENCE_CLASS_SEPARATION",
    }
    write_json(OUTPUT / "step29n_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
