import importlib.util
import inspect
import json
from pathlib import Path
import unittest


WORKSPACE = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "step28n", WORKSPACE / "scripts" / "integrate_path_block_persistence_policy.py"
)
STEP28N = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STEP28N)


class PathBlockPersistenceIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = STEP28N.read_json(STEP28N.OUTPUT / "step19_contract_audit.json")
        cls.control_input = STEP28N.read_json(STEP28N.OUTPUT / "control_policy_input.json")
        cls.control_result = STEP28N.read_json(STEP28N.OUTPUT / "control_policy_result.json")
        cls.blocked_input = STEP28N.read_json(STEP28N.OUTPUT / "blocked_policy_input.json")
        cls.sequence = STEP28N.read_json(STEP28N.OUTPUT / "policy_sequence.json")
        cls.trace = STEP28N.read_json(STEP28N.OUTPUT / "policy_transition_trace.json")["observations"]
        cls.decision = STEP28N.read_json(STEP28N.OUTPUT / "persistence_decision.json")
        cls.boundary = STEP28N.read_json(STEP28N.OUTPUT / "replan_decision_boundary.json")
        cls.summary = STEP28N.read_json(STEP28N.OUTPUT / "step28n_summary.json")

    def test_01_step27n_artifacts_load(self):
        summary, comparison, _, _, _ = STEP28N.load_step27_evidence()
        self.assertEqual(summary["step_result"], "PASS")
        self.assertEqual(comparison["positive_case"]["after"], "BLOCKED")

    def test_02_case_a_provenance_preserved(self):
        self.assertEqual(self.control_input["evidence_class"], "OBSERVED_NAVIGATION_APPROXIMATION")

    def test_03_case_a_valid_status_preserved(self):
        self.assertEqual(self.control_input["path_validity"], "VALID")
        self.assertFalse(self.control_input["path_blocked"])

    def test_04_case_a_does_not_trigger_blocking_policy(self):
        self.assertEqual(self.control_result["navigation_state"], "MOVING")
        self.assertEqual(self.control_result["action"], "CONTINUE_CURRENT_PATH")
        self.assertEqual(self.control_result["replan_status"], "NOT_RUN")

    def test_05_case_b_provenance_preserved(self):
        self.assertEqual(self.blocked_input["evidence_class"], "DOCUMENTED_DEVELOPMENT_FIXTURE")

    def test_06_case_b_blocked_status_preserved(self):
        self.assertEqual(self.blocked_input["path_validity"], "BLOCKED")
        self.assertTrue(self.blocked_input["path_blocked"])

    def test_07_dynamic_box_identity_preserved(self):
        self.assertEqual(self.blocked_input["obstacle_id"], "dynamic_box_02")
        self.assertTrue(all(item["raw_detected_obstacle_ids"] == ["dynamic_box_02"] for item in self.trace))

    def test_08_path_identity_preserved(self):
        expected = "17646cb8b5893a1c099c67ea2700c4fcd0a9367d13b1d83a19e705f2d19dcdc7"
        self.assertEqual(self.control_input["path_sha256"], expected)
        self.assertEqual(self.blocked_input["path_sha256"], expected)

    def test_09_blocking_waypoint_provenance_preserved(self):
        self.assertEqual(self.blocked_input["blocking_provenance"]["first_blocked_waypoint_index"], 50)

    def test_10_blocking_edge_provenance_preserved(self):
        self.assertEqual(self.blocked_input["blocking_provenance"]["first_blocked_edge_index"], 49)

    def test_11_existing_step19_input_contract_used(self):
        self.assertEqual(
            self.contract["input_schema"],
            ["observation_index", "logical_time_s", "robot_waypoint_index", "detected_obstacle_ids"],
        )
        self.assertIn("evaluate_policy_observation", inspect.getsource(STEP28N))

    def test_12_existing_step19_parameters_reused(self):
        self.assertEqual(self.contract["parameters"]["presence_confirmation_count"], 2)
        self.assertEqual(self.contract["parameters"]["absence_confirmation_count"], 2)
        self.assertEqual(self.contract["parameters"]["persistence_threshold_s"], 1.0)
        self.assertEqual(self.contract["parameters"]["observation_interval_s"], 0.2)

    def test_13_threshold_config_unmodified(self):
        self.assertEqual(STEP28N.sha256_file(STEP28N.POLICY_CONFIG), self.contract["source_of_truth"]["config_sha256"])

    def test_14_single_block_not_manually_promoted(self):
        self.assertEqual(self.trace[1]["classification"], "TEMPORARY_BLOCKING")
        self.assertFalse(self.decision["single_block_sample_manually_promoted"])

    def test_15_presence_confirmation_behavior(self):
        self.assertEqual(self.trace[0]["temporal_obstacle_states"]["dynamic_box_02"]["state"], "PRESENCE_PENDING")
        self.assertEqual(self.trace[1]["temporal_obstacle_states"]["dynamic_box_02"]["state"], "CONFIRMED_PRESENT")

    def test_16_existing_persistence_behavior(self):
        self.assertEqual(self.trace[-1]["blocking_duration_s"], 1.0)
        self.assertEqual(self.trace[-1]["classification"], "PERSISTENT_BLOCKING")

    def test_17_existing_absence_confirmation_behavior(self):
        reference = self.sequence["absence_resume_reference"]
        p1 = STEP28N.read_json(reference["source"])
        self.assertEqual(reference["confirmation_count"], 2)
        self.assertEqual(p1["obstacle_trackers"]["p1_box"]["clear_confirmation_history"][0]["time_to_confirmation_s"], 0.2)

    def test_18_stop_transition_follows_policy(self):
        self.assertEqual((self.trace[1]["navigation_state"], self.trace[1]["action"]), ("STOPPED_FOR_OBSTACLE", "STOP_AND_WAIT"))

    def test_19_wait_transition_follows_policy(self):
        self.assertEqual((self.trace[2]["navigation_state"], self.trace[2]["action"]), ("WAITING_FOR_CLEARANCE", "STOP_AND_WAIT"))

    def test_20_documented_resume_behavior_preserved(self):
        reference = self.sequence["absence_resume_reference"]
        self.assertEqual(reference["documented_action"], "RESUME_CURRENT_PATH")
        self.assertFalse(reference["case_b_disappearance_executed"])

    def test_21_replan_required_decision_reached(self):
        self.assertEqual(self.decision["replan_status"], "REPLAN_REQUIRED")
        self.assertEqual(self.boundary["decision"], "REPLAN_REQUIRED")

    def test_22_policy_decision_does_not_invoke_step18(self):
        self.assertEqual(self.boundary["step18_handoff"], "NOT_INVOKED")
        self.assertIsNone(self.trace[-1]["replan_result"])
        self.assertEqual(self.summary["step18_call"], "NOT_RUN")

    def test_23_no_astar_execution(self):
        self.assertEqual(self.summary["astar"], "NOT_RUN")
        self.assertNotIn("plan_path", inspect.getsource(STEP28N))

    def test_24_no_new_path_generation(self):
        self.assertEqual(self.boundary["new_path_generation"], "NOT_RUN")
        self.assertEqual(self.boundary["active_path_revision"], 0)
        self.assertEqual(self.boundary["planner_runs"], 0)

    def test_25_no_qwen_inference(self):
        self.assertEqual(self.summary["qwen_inference_runs"], 0)

    def test_26_no_homography_recomputation(self):
        self.assertEqual(self.summary["homography_recomputation"], "NOT_RUN")

    def test_27_step23m_unchanged(self):
        measurement = STEP28N.read_json(STEP28N.STEP23M)
        self.assertEqual(measurement["measurement_input_status"], "WAITING_FOR_USER_MEASUREMENT")

    def test_28_physical_geometry_unused(self):
        self.assertEqual(self.summary["physical_geometry"], "NOT_READY_NOT_USED")

    def test_29_observed_and_development_evidence_separated(self):
        self.assertNotEqual(self.control_input["evidence_class"], self.blocked_input["evidence_class"])
        self.assertFalse(self.blocked_input["camera_or_vlm_temporal_evidence"])
        self.assertTrue(self.summary["observed_and_development_evidence_separate"])

    def test_30_hardware_stop_not_claimed(self):
        self.assertEqual(self.boundary["hardware_stop"], "NOT_CLAIMED")
        self.assertFalse(self.summary["hardware_stop_verified"])

    def test_31_step27_artifacts_unchanged(self):
        self.assertEqual(self.summary["step27_artifacts_before"], self.summary["step27_artifacts_after"])
        self.assertTrue(self.summary["step27_artifacts_unchanged"])

    def test_32_protected_artifacts_unchanged(self):
        self.assertEqual(STEP28N.snapshot_protected(), STEP28N.PROTECTED_BASELINE)
        self.assertTrue(self.summary["protected_artifacts_unchanged"])


if __name__ == "__main__":
    unittest.main()
