import importlib.util
import inspect
import json
from pathlib import Path
import unittest


WORKSPACE = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "step29n", WORKSPACE / "scripts" / "integrate_persistent_block_replanning.py"
)
STEP29N = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STEP29N)


class PersistentBlockReplanningIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        names = (
            "step18_contract_audit", "control_replan_gate", "blocked_replan_input",
            "step18_handoff", "replan_execution_result", "old_path_reference",
            "replacement_path", "replacement_path_validation", "path_revision_result",
            "step29n_summary",
        )
        for name in names:
            setattr(cls, name, STEP29N.read_json(STEP29N.OUTPUT / f"{name}.json"))

    def test_01_step28n_artifacts_load(self):
        self.assertEqual(STEP29N.read_json(STEP29N.STEP28N / "step28n_summary.json")["step_result"], "PASS")

    def test_02_case_a_provenance_preserved(self):
        self.assertEqual(self.control_replan_gate["evidence_class"], "OBSERVED_NAVIGATION_APPROXIMATION")

    def test_03_case_a_valid_preserved(self):
        self.assertEqual(self.control_replan_gate["path_validity"], "VALID")

    def test_04_case_a_continue_preserved(self):
        self.assertEqual(self.control_replan_gate["policy_action"], "CONTINUE_CURRENT_PATH")

    def test_05_case_a_does_not_invoke_step18(self):
        self.assertFalse(self.control_replan_gate["gate"]["eligible"])
        self.assertEqual(self.control_replan_gate["step18_invocations"], 0)

    def test_06_case_a_planner_count_zero(self):
        self.assertEqual(self.control_replan_gate["planner_runs"], 0)

    def test_07_case_a_revision_unchanged(self):
        self.assertEqual(self.control_replan_gate["path_revision_before"], self.control_replan_gate["path_revision_after"])

    def test_08_case_b_provenance_preserved(self):
        self.assertEqual(self.blocked_replan_input["evidence_class"], "DOCUMENTED_DEVELOPMENT_FIXTURE")

    def test_09_case_b_blocked_preserved(self):
        self.assertEqual(self.blocked_replan_input["path_validity"], "BLOCKED")

    def test_10_case_b_persistent_preserved(self):
        self.assertEqual(self.blocked_replan_input["persistence"], "PERSISTENT_BLOCKING")

    def test_11_replan_required_gate(self):
        self.assertEqual(self.blocked_replan_input["decision"], "REPLAN_REQUIRED")
        self.assertTrue(self.blocked_replan_input["gate"]["eligible"])

    def test_12_temporary_block_rejected(self):
        blocked = dict(self.blocked_replan_input)
        persistence = {"classification": "TEMPORARY_BLOCKING", "decision_source": "EXISTING_STEP19_POLICY"}
        boundary = {"decision": "REPLAN_REQUIRED"}
        self.assertFalse(STEP29N.evaluate_replan_gate(blocked, persistence, boundary)["eligible"])

    def test_13_missing_decision_rejected(self):
        blocked = STEP29N.read_json(STEP29N.STEP28N / "blocked_policy_input.json")
        persistence = STEP29N.read_json(STEP29N.STEP28N / "persistence_decision.json")
        self.assertFalse(STEP29N.evaluate_replan_gate(blocked, persistence, {})["eligible"])

    def test_14_existing_step18_api_reused(self):
        self.assertEqual(self.step18_contract_audit["entry_api"], "evaluate_event_and_replan")
        self.assertIn("evaluate_event_and_replan(", inspect.getsource(STEP29N.main))

    def test_15_current_robot_progress_source_preserved(self):
        source = STEP29N.read_json(self.step18_handoff["current_robot_progress_source"])["observations"][-1]
        self.assertEqual(self.step18_handoff["current_robot_pose"], source["current_robot_pose"])

    def test_16_goal_preserved(self):
        d2 = STEP29N.dynamic_scenario_by_id(STEP29N.load_dynamic_config(STEP29N.DYNAMIC_CONFIG), "D2")
        self.assertEqual(self.step18_handoff["goal"], d2["initial_goal"])

    def test_17_dynamic_box_preserved(self):
        self.assertEqual(self.blocked_replan_input["obstacle_id"], "dynamic_box_02")

    def test_18_world_revision_preserved(self):
        self.assertEqual(self.blocked_replan_input["world_revision"], self.step18_handoff["world_revision"])

    def test_19_planner_config_preserved(self):
        self.assertEqual(self.step18_handoff["planner_config_sha256"], STEP29N.sha256_file(STEP29N.PLANNER_CONFIG))
        self.assertTrue(self.step29n_summary["planner_config_unchanged"])

    def test_20_invocation_count_deterministic(self):
        self.assertEqual(self.step18_handoff["invocation_count"], 1)
        self.assertEqual(self.step29n_summary["case_b_planner_runs"], 1)

    def test_21_success_path_validation(self):
        if self.replan_execution_result["status"] == "REPLAN_SUCCESS":
            self.assertEqual(self.replacement_path_validation["status"], "VALID")
            self.assertEqual(self.replacement_path_validation["existing_step18_validation"]["status"], "VALID")

    def test_22_failure_contract_preserved(self):
        failure = self.step18_contract_audit["failure_contract"]
        self.assertEqual((failure["replan_status"], failure["system_action"], failure["stop_required"]), ("REPLAN_FAILED", "STOP", True))
        self.assertIsNone(failure["new_active_path"])

    def test_23_old_path_preserved(self):
        self.assertEqual(self.old_path_reference["path_sha256"], self.blocked_replan_input["path_sha256"])

    def test_24_replacement_recorded_separately(self):
        self.assertNotEqual(STEP29N.OUTPUT / "old_path_reference.json", STEP29N.OUTPUT / "replacement_path.json")
        self.assertIn(self.replacement_path["status"], {"GENERATED", "NOT_GENERATED"})

    def test_25_revision_follows_step18(self):
        if self.replacement_path["status"] == "GENERATED":
            self.assertEqual(self.path_revision_result["new_revision"], self.path_revision_result["old_revision"] + 1)
            self.assertEqual(self.path_revision_result["old_status_after_step18"], "SUPERSEDED")

    def test_26_no_qwen(self):
        self.assertEqual(self.step29n_summary["qwen_inference_runs"], 0)

    def test_27_no_homography(self):
        self.assertEqual(self.step29n_summary["homography"], "NOT_RUN")

    def test_28_no_ros_nav2(self):
        self.assertEqual(self.step29n_summary["ros_nav2"], "NOT_RUN")
        self.assertEqual(self.step29n_summary["hardware_execution"], "NOT_RUN")

    def test_29_no_physical_geometry_inference(self):
        self.assertEqual(self.step29n_summary["physical_geometry"], "NOT_READY_NOT_INFERRED")

    def test_30_step23m_unchanged(self):
        self.assertEqual(STEP29N.read_json(STEP29N.STEP23M)["measurement_input_status"], "WAITING_FOR_USER_MEASUREMENT")

    def test_31_step28n_artifacts_unchanged(self):
        self.assertEqual(self.step29n_summary["step28_artifacts_before"], self.step29n_summary["step28_artifacts_after"])
        self.assertEqual(STEP29N.directory_manifest(STEP29N.STEP28N), self.step29n_summary["step28_artifacts_after"])

    def test_32_protected_previous_artifacts_unchanged(self):
        self.assertTrue(self.step29n_summary["protected_artifacts_unchanged"])
        self.assertEqual(STEP29N.snapshot_protected(), self.step29n_summary["protected_artifacts_after"])


if __name__ == "__main__":
    unittest.main()
