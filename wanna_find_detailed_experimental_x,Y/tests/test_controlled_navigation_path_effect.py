import importlib.util
import inspect
import json
from pathlib import Path
import unittest


WORKSPACE = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "step27n", WORKSPACE / "scripts" / "run_controlled_navigation_path_effect.py"
)
STEP27N = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STEP27N)


class ControlledNavigationPathEffectTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.control = STEP27N.read_json(STEP27N.OUTPUT / "control_case_reference.json")
        cls.audit = STEP27N.read_json(STEP27N.OUTPUT / "fixture_audit.json")
        cls.selection = STEP27N.read_json(STEP27N.OUTPUT / "positive_fixture_selection.json")
        cls.before = STEP27N.read_json(STEP27N.OUTPUT / "positive_before_state.json")
        cls.after = STEP27N.read_json(STEP27N.OUTPUT / "positive_after_state.json")
        cls.before_validity = STEP27N.read_json(STEP27N.OUTPUT / "positive_path_validity_before.json")
        cls.after_validity = STEP27N.read_json(STEP27N.OUTPUT / "positive_path_validity_after.json")
        cls.blocking = STEP27N.read_json(STEP27N.OUTPUT / "positive_blocking_provenance.json")
        cls.comparison = STEP27N.read_json(STEP27N.OUTPUT / "controlled_effect_comparison.json")
        cls.summary = STEP27N.read_json(STEP27N.OUTPUT / "step27n_summary.json")

    def test_01_control_is_observed_case(self):
        self.assertEqual(self.control["provenance_class"], "OBSERVED_NAVIGATION_APPROXIMATION")

    def test_02_control_identity_is_preserved(self):
        self.assertEqual(self.control["observation_id"], "image_014_photo_08_track_039")

    def test_03_control_path_identity_is_preserved(self):
        self.assertEqual(self.control["path_sha256"], "17646cb8b5893a1c099c67ea2700c4fcd0a9367d13b1d83a19e705f2d19dcdc7")

    def test_04_control_before_is_valid(self):
        self.assertEqual(self.control["before"], "VALID")

    def test_05_control_after_is_valid(self):
        self.assertEqual(self.control["after"], "VALID")

    def test_06_control_has_no_path_effect(self):
        self.assertEqual(self.control["effect"], "NO_PATH_EFFECT")

    def test_07_existing_fixture_audit_finds_positive_cases(self):
        self.assertTrue(any(item["qualifies_as_positive_fixture"] for item in self.audit["audited"]))

    def test_08_documented_d2_fixture_is_selected(self):
        self.assertEqual((self.selection["scenario_id"], self.selection["event_id"]), ("D2", "D2_add_on_path"))

    def test_09_positive_path_existed_before_step27n(self):
        self.assertTrue(self.selection["path_existed_before_step27n"])
        self.assertTrue(Path(self.selection["source_artifact"]).is_file())

    def test_10_positive_same_path_before_after(self):
        case = self.comparison["positive_case"]
        self.assertEqual(case["path_sha256_before"], case["path_sha256_after"])
        self.assertTrue(self.summary["positive_path_same_before_after"])

    def test_11_positive_before_is_valid(self):
        self.assertEqual(self.before_validity["path_state"], "VALID")

    def test_12_positive_after_is_blocked(self):
        self.assertEqual(self.after_validity["path_state"], "BLOCKED")

    def test_13_first_blocking_location_is_recorded(self):
        self.assertEqual(self.blocking["first_blocked_waypoint_index"], 50)
        self.assertEqual(self.blocking["first_blocked_edge_index"], 49)
        self.assertEqual(self.blocking["blocking_grid_cell"], [36, 46])

    def test_14_blocker_identity_is_recorded(self):
        self.assertEqual(self.blocking["blocking_obstacle_ids"], ["dynamic_box_02"])

    def test_15_blocking_cell_changed_to_occupied(self):
        self.assertEqual((self.blocking["blocking_cell_occupancy_before"], self.blocking["blocking_cell_occupancy_after"]), (0, 1))
        self.assertTrue(self.blocking["blocking_cell_newly_occupied"])

    def test_16_evidence_classes_remain_separate(self):
        self.assertEqual(self.comparison["positive_case"]["provenance"], "DOCUMENTED_DEVELOPMENT_FIXTURE")
        self.assertFalse(self.comparison["interpretation"]["evidence_classes_combined"])

    def test_17_positive_fixture_is_not_camera_or_vlm_evidence(self):
        self.assertFalse(self.selection["camera_or_vlm_derived"])

    def test_18_existing_fixture_and_config_are_unmodified(self):
        self.assertEqual(STEP27N.sha256_file(self.selection["source_artifact"]), self.selection["source_artifact_sha256"])
        self.assertEqual(STEP27N.sha256_file(self.selection["config_source"]), self.selection["config_source_sha256"])
        self.assertFalse(self.selection["fixture_modified"])

    def test_19_no_physical_geometry_inference(self):
        self.assertEqual(self.summary["physical_geometry"], "NOT_READY_NOT_USED")
        self.assertEqual(self.summary["step23m_measurement_input_status"], "WAITING_FOR_USER_MEASUREMENT")

    def test_20_existing_grid_and_safety_policy_are_reused(self):
        self.assertEqual(self.before["collision_world"]["safety_margin_m"], 0.0)
        self.assertEqual(self.after["collision_world"]["safety_margin_m"], 0.0)

    def test_21_no_astar_or_new_path_generation(self):
        self.assertEqual(self.summary["astar"], "NOT_RUN")
        self.assertEqual(self.summary["new_path_generation"], "NOT_RUN")
        self.assertNotIn("plan_path", inspect.getsource(STEP27N))

    def test_22_no_step18_execution(self):
        self.assertEqual(self.summary["step18_call"], "NOT_RUN")

    def test_23_no_step19_execution(self):
        self.assertEqual(self.summary["step19_call"], "NOT_RUN")

    def test_24_no_qwen_inference(self):
        self.assertEqual(self.summary["qwen_inference_runs"], 0)

    def test_25_no_homography_recomputation(self):
        self.assertEqual(self.summary["homography_recomputation"], "NOT_RUN")

    def test_26_no_replanning_or_ros_execution(self):
        self.assertEqual(self.summary["replanning"], "NOT_RUN")
        self.assertEqual(self.summary["ros_nav2"], "NOT_RUN")

    def test_27_step26n_control_artifacts_are_unchanged(self):
        self.assertEqual(self.summary["control_artifacts_before"], self.summary["control_artifacts_after"])
        self.assertTrue(self.summary["control_artifacts_unchanged"])

    def test_28_protected_artifacts_are_unchanged(self):
        self.assertEqual(STEP27N.snapshot_protected(), STEP27N.PROTECTED_BASELINE)
        self.assertTrue(self.summary["protected_artifacts_unchanged"])


if __name__ == "__main__":
    unittest.main()
