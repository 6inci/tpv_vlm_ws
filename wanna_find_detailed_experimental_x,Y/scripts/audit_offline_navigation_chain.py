#!/usr/bin/env python3
"""Audit the two existing offline navigation evidence tracks without rerunning them."""

import hashlib
import json
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
RESULTS = WORKSPACE / "results"
STEP22D = RESULTS / "candidate_conditioned_vlm_revalidation_method_d"
STEP24N = RESULTS / "navigation_obstacle_adapter"
STEP25N = RESULTS / "navigation_dynamic_world_integration"
STEP26N = RESULTS / "navigation_path_validity"
STEP27N = RESULTS / "controlled_navigation_path_effect"
STEP28N = RESULTS / "path_block_persistence_integration"
STEP29N = RESULTS / "persistent_block_replanning_integration"
STEP23M = RESULTS / "physical_measurement_registration" / "measurement_registration_summary.json"
OUTPUT = RESULTS / "offline_navigation_chain_integration"


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def artifact_ref(path):
    path = Path(path)
    return {"path": str(path), "sha256": sha256_file(path)}


def snapshot_protected():
    later_result = RESULTS / "observed_blocking_acquisition"
    records = [
        (str(path), sha256_file(path))
        for path in sorted(RESULTS.rglob("*"))
        if path.is_file() and OUTPUT not in path.parents and later_result not in path.parents
    ]
    return {
        "file_count": len(records),
        "digest_sha256": hashlib.sha256(
            "".join(f"{file_hash}  {path}\n" for path, file_hash in records).encode()
        ).hexdigest(),
    }


def load_sources():
    return {
        "s22": read_json(STEP22D / "step22d_summary.json"),
        "semantic": read_json(STEP22D / "parsed_result.json")["semantic_result"],
        "association": read_json(STEP22D / "association_result.json"),
        "metric": read_json(STEP22D / "metric_result.json"),
        "s24": read_json(STEP24N / "step24n_summary.json"),
        "observation": read_json(STEP24N / "navigation_obstacle_observation.json"),
        "s25": read_json(STEP25N / "step25n_summary.json"),
        "dynamic": read_json(STEP25N / "dynamic_obstacle_record.json"),
        "s26": read_json(STEP26N / "step26n_summary.json"),
        "effect": read_json(STEP26N / "path_effect_classification.json"),
        "comparison": read_json(STEP27N / "controlled_effect_comparison.json"),
        "fixture": read_json(STEP27N / "positive_fixture_selection.json"),
        "blocked": read_json(STEP28N / "blocked_policy_input.json"),
        "persistence": read_json(STEP28N / "persistence_decision.json"),
        "boundary": read_json(STEP28N / "replan_decision_boundary.json"),
        "control_gate": read_json(STEP29N / "control_replan_gate.json"),
        "blocked_gate": read_json(STEP29N / "blocked_replan_input.json"),
        "handoff": read_json(STEP29N / "step18_handoff.json"),
        "execution": read_json(STEP29N / "replan_execution_result.json"),
        "replacement": read_json(STEP29N / "replacement_path.json"),
        "validation": read_json(STEP29N / "replacement_path_validation.json"),
        "revision": read_json(STEP29N / "path_revision_result.json"),
        "s29": read_json(STEP29N / "step29n_summary.json"),
        "measurement": read_json(STEP23M),
    }


def consistency(s):
    world_xy = s["metric"]["world_xy"]
    observation = s["observation"]
    dynamic = s["dynamic"]
    control = s["comparison"]["control_case"]
    positive = s["comparison"]["positive_case"]
    checks = {
        "track_a_step22d_pass": s["s22"]["step_result"] == "PASS",
        "track_a_image_id": s["s22"]["image_id"] == observation["source_image_id"] == dynamic["observation_identity"]["source_image_id"],
        "track_a_track_id": s["s22"]["track_id"] == observation["track_id"] == dynamic["observation_identity"]["track_id"],
        "track_a_label": s["semantic"]["visible_label"] == observation["semantic"]["label"] == "tennis racket",
        "track_a_semantic_status": s["semantic"]["status"] == observation["semantic"]["status"] == "VISIBLE_ENTITY_FOUND",
        "track_a_association": s["association"]["status"] == observation["evidence"]["association"] == "ASSOCIATED",
        "track_a_world_reference": world_xy == observation["reference_point_world"]["xy_m"] == dynamic["reference_point_world"]["xy_m"],
        "track_a_coordinate_frame": s["metric"]["coordinate_frame"] == observation["reference_point_world"]["frame"] == dynamic["coordinate_frame"],
        "track_a_navigation_role": observation["navigation_role"] == s["s24"]["navigation_role"] == "OBSTACLE_CANDIDATE",
        "track_a_cell": observation["navigation_representation"]["grid_index"] == {
            "row": dynamic["navigation_cell"]["row"], "col": dynamic["navigation_cell"]["col"]
        },
        "track_a_dynamic_identity": s["s25"]["dynamic_obstacle_id"] == s["s26"]["dynamic_obstacle_id"] == s["effect"]["dynamic_obstacle_id"],
        "track_a_path_result": s["s26"]["path_validity_after"] == control["after"] == s["control_gate"]["path_validity"] == "VALID",
        "track_a_no_path_effect": s["effect"]["classification"] == control["effect"] == "NO_PATH_EFFECT",
        "track_a_no_replan": s["control_gate"]["step18_invocations"] == s["s29"]["case_a_step18_invocations"] == 0,
        "track_b_fixture": s["fixture"]["scenario_id"] == s["blocked"]["scenario_id"] == "D2",
        "track_b_obstacle": positive["blocking_obstacle_ids"] == s["blocked"]["blocking_provenance"]["blocking_obstacle_ids"] == ["dynamic_box_02"],
        "track_b_path_identity": positive["path_sha256_after"] == s["blocked"]["path_sha256"] == s["blocked_gate"]["path_sha256"],
        "track_b_blocked": positive["after"] == s["blocked"]["path_validity"] == "BLOCKED",
        "track_b_persistent": s["persistence"]["classification"] == s["blocked_gate"]["persistence"] == "PERSISTENT_BLOCKING",
        "track_b_replan_required": s["boundary"]["decision"] == s["blocked_gate"]["decision"] == "REPLAN_REQUIRED",
        "track_b_step18_invocation": s["handoff"]["invocation_count"] == s["s29"]["case_b_step18_invocations"] == 1,
        "track_b_replan_success": s["execution"]["status"] == s["s29"]["case_b_replanning_result"] == "REPLAN_SUCCESS",
        "track_b_replacement_identity": s["replacement"]["path"]["path_id"] == s["revision"]["new_path_id"] == "D2_path_r1",
        "track_b_replacement_valid": s["validation"]["status"] == "VALID",
        "track_b_revision": s["revision"]["old_revision"] == 0 and s["revision"]["new_revision"] == 1,
        "evidence_classes_separate": control["provenance"] == "OBSERVED_NAVIGATION_APPROXIMATION" and positive["provenance"] == "DOCUMENTED_DEVELOPMENT_FIXTURE",
        "physical_measurement_waiting": s["measurement"]["measurement_input_status"] == "WAITING_FOR_USER_MEASUREMENT",
    }
    if not all(checks.values()):
        raise RuntimeError(f"cross-step consistency failure: {[key for key, value in checks.items() if not value]}")
    return checks


def main():
    protected_before = snapshot_protected()
    s = load_sources()
    checks = consistency(s)
    refs = {
        "step22d": artifact_ref(STEP22D / "step22d_summary.json"),
        "step24n": artifact_ref(STEP24N / "navigation_obstacle_observation.json"),
        "step25n": artifact_ref(STEP25N / "dynamic_obstacle_record.json"),
        "step26n": artifact_ref(STEP26N / "step26n_summary.json"),
        "step27n": artifact_ref(STEP27N / "controlled_effect_comparison.json"),
        "step28n": artifact_ref(STEP28N / "persistence_decision.json"),
        "step29n": artifact_ref(STEP29N / "step29n_summary.json"),
    }
    protected_after = snapshot_protected()
    if protected_before != protected_after:
        raise RuntimeError("prior result artifacts changed during STEP 30N audit")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT / "observed_track_lineage.json", {
        "track": "TRACK_A_OBSERVED_CONTROL",
        "evidence_class": "OBSERVED_NAVIGATION_APPROXIMATION",
        "camera_vlm_derived": True,
        "source": {
            "image_id": s["s22"]["image_id"], "track_id": s["s22"]["track_id"],
            "label": s["semantic"]["visible_label"], "semantic_status": s["semantic"]["status"],
            "association": s["association"]["status"], "ground_reference": s["metric"]["ground_reference"]["meaning"],
        },
        "world_reference": {"xy_m": s["metric"]["world_xy"], "frame": s["metric"]["coordinate_frame"]},
        "navigation": {
            "role": s["observation"]["navigation_role"],
            "geometry": s["observation"]["navigation_representation"]["geometry_status"],
            "cell": s["observation"]["navigation_representation"]["grid_index"],
            "dynamic_obstacle_id": s["s25"]["dynamic_obstacle_id"],
            "path_validity": s["s26"]["path_validity_after"],
            "path_effect": s["s26"]["effect_classification"],
        },
        "policy": s["control_gate"]["policy_action"],
        "step18_invocations": s["control_gate"]["step18_invocations"],
        "replacement_path": s["control_gate"]["new_path"],
        "physical_geometry": "NOT_READY_NOT_USED",
        "artifact_references": {key: refs[key] for key in ("step22d", "step24n", "step25n", "step26n", "step27n", "step29n")},
    })
    write_json(OUTPUT / "development_track_lineage.json", {
        "track": "TRACK_B_DEVELOPMENT_POSITIVE",
        "evidence_class": "DOCUMENTED_DEVELOPMENT_FIXTURE",
        "camera_vlm_derived": False,
        "fixture": {"scenario_id": s["fixture"]["scenario_id"], "event_id": s["fixture"]["event_id"], "obstacle_id": s["blocked"]["obstacle_id"]},
        "path": {"sha256": s["blocked"]["path_sha256"], "validity": s["blocked"]["path_validity"]},
        "persistence": {"classification": s["persistence"]["classification"], "decision": s["boundary"]["decision"], "decision_source": s["persistence"]["decision_source"]},
        "replanning": {
            "implementation": s["handoff"]["implementation"], "invocations": s["handoff"]["invocation_count"],
            "current_robot_pose": s["handoff"]["current_robot_pose"], "goal": s["handoff"]["goal"],
            "world_revision": s["handoff"]["world_revision"], "result": s["execution"]["status"],
        },
        "old_path": {"waypoint_count": s["s29"]["old_path_waypoint_count"], "length_m": s["s29"]["old_path_length_m"], "revision": s["revision"]["old_revision"]},
        "replacement_path": {"path_id": s["revision"]["new_path_id"], "waypoint_count": s["s29"]["new_path_waypoint_count"], "length_m": s["s29"]["new_path_length_m"], "revision": s["revision"]["new_revision"], "validation": s["validation"]["status"]},
        "artifact_references": {key: refs[key] for key in ("step27n", "step28n", "step29n")},
    })
    write_json(OUTPUT / "cross_step_consistency.json", {"status": "PASS", "checks": checks, "artifact_references": refs})
    write_json(OUTPUT / "evidence_class_separation.json", {
        "status": "PRESERVED",
        "tracks_combined": False,
        "observed_track": {"evidence_class": "OBSERVED_NAVIGATION_APPROXIMATION", "camera_vlm_derived": True, "blocking_observed": False},
        "development_track": {"evidence_class": "DOCUMENTED_DEVELOPMENT_FIXTURE", "camera_vlm_derived": False, "blocking_observed": True},
        "prohibited_claim": "THE_OBSERVED_TENNIS_RACKET_CAUSED_REPLANNING",
        "supported_statement": "An actual camera/VLM-derived observation entered navigation and correctly produced no path effect; a separate documented blocking fixture verified blocking, persistence, and replanning.",
    })
    write_json(OUTPUT / "system_capability_matrix.json", {
        "supported": {
            "semantic_metric_to_navigation_without_physical_dimensions": True,
            "unaffected_path_remains_valid": True,
            "dynamic_obstacle_presence_does_not_force_replanning": True,
            "documented_block_detection": True,
            "persistent_block_gates_step18": True,
            "existing_step18_generates_valid_replacement": True,
            "cross_module_provenance_traceable": True,
        },
        "not_supported": {
            "observed_camera_obstacle_caused_replanning": "NOT_SUPPORTED",
            "live_temporal_persistence": "NOT_SUPPORTED",
            "physical_robot_replanning": "NOT_SUPPORTED",
            "real_world_collision_avoidance_performance": "NOT_SUPPORTED",
            "physical_obstacle_footprint_accuracy": "NOT_SUPPORTED",
            "ros_nav2_end_to_end_execution": "NOT_SUPPORTED",
        },
    })
    write_json(OUTPUT / "system_limitations.json", {
        "observed_blocking_example": "NOT_AVAILABLE",
        "physical_measurement": s["measurement"]["measurement_input_status"],
        "physical_geometry": "NOT_READY",
        "live_camera_temporal_evidence": "NOT_AVAILABLE",
        "qwen_inference_runs_step30n": 0,
        "homography_recomputation": "NOT_RUN",
        "ros_nav2": "NOT_RUN",
        "hardware_execution": "NOT_RUN",
        "safety_claim": "DEVELOPMENT_SOFTWARE_EVIDENCE_ONLY",
    })
    write_json(OUTPUT / "step30n_summary.json", {
        "schema_version": "0.1", "step_result": "PASS",
        "observed_control_track": {
            "source": "OBSERVED_NAVIGATION_APPROXIMATION", "semantic_metric": "PASS",
            "navigation_adapter": "PASS", "dynamic_world": "REGISTERED",
            "path_validity": "VALID", "path_effect": "NO_PATH_EFFECT",
            "policy": "CONTINUE_CURRENT_PATH", "replanning": "NOT_TRIGGERED",
        },
        "development_positive_track": {
            "source": "DOCUMENTED_DEVELOPMENT_FIXTURE", "path_validity": "BLOCKED",
            "persistence": "PERSISTENT_BLOCKING", "decision": "REPLAN_REQUIRED",
            "replanning": "REPLAN_SUCCESS", "replacement_validation": "VALID",
        },
        "evidence_classes_separate": True,
        "qwen_inference_runs": 0, "homography_recomputation": "NOT_RUN",
        "ros_nav2": "NOT_RUN", "hardware_execution": "NOT_RUN",
        "physical_geometry": "NOT_READY_NOT_INFERRED",
        "step23m_measurement_input_status": s["measurement"]["measurement_input_status"],
        "protected_artifacts_before": protected_before,
        "protected_artifacts_after": protected_after,
        "protected_artifacts_unchanged": protected_before == protected_after,
        "verification": {},
        "recommended_next_step": "CONTROLLED_OBSERVED_BLOCKING_DATA_ACQUISITION_EXPERIMENT",
    })
    print(json.dumps(read_json(OUTPUT / "step30n_summary.json"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
