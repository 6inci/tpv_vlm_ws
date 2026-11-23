import importlib.util
import inspect
import json
import math
from pathlib import Path
import sys
import unittest


WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))
SPEC = importlib.util.spec_from_file_location(
    "step26n", WORKSPACE / "scripts" / "evaluate_navigation_path_validity.py"
)
STEP26N = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STEP26N)

from vlm_reposition.planning import validate_current_path


class NavigationPathValidityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.step25, cls.before_world, cls.after_world, cls.obstacle, cls.occupancy = STEP26N.load_step25n_state()
        cls.path_source, cls.path, cls.path_digest = STEP26N.load_existing_path()
        cls.before_path = STEP26N.read_json(STEP26N.OUTPUT / "path_before.json")
        cls.after_path = STEP26N.read_json(STEP26N.OUTPUT / "path_after.json")
        cls.before_validity = STEP26N.read_json(STEP26N.OUTPUT / "path_validity_before.json")
        cls.after_validity = STEP26N.read_json(STEP26N.OUTPUT / "path_validity_after.json")
        cls.provenance = STEP26N.read_json(STEP26N.OUTPUT / "blocking_provenance.json")
        cls.summary = STEP26N.read_json(STEP26N.OUTPUT / "step26n_summary.json")

    def test_01_step25n_artifacts_load(self):
        self.assertEqual(self.step25["step_result"], "PASS")
        self.assertEqual(self.step25["dynamic_obstacle"], "REGISTERED")

    def test_02_dynamic_obstacle_id_preserved(self):
        expected = "dynamic_image_014_photo_08_track_039"
        self.assertEqual(self.obstacle["id"], expected)
        self.assertEqual(self.summary["dynamic_obstacle_id"], expected)

    def test_03_before_world_revision(self):
        self.assertEqual(self.before_world["obstacle_state"]["world_revision"], 0)
        self.assertEqual(self.before_validity["world_revision"], 0)

    def test_04_after_world_revision(self):
        self.assertEqual(self.after_world["obstacle_state"]["world_revision"], 1)
        self.assertEqual(self.after_validity["world_revision"], 1)

    def test_05_existing_path_source_identified(self):
        self.assertEqual(self.path_source["scenario_id"], "D1")
        self.assertEqual(self.summary["path_source"], "STEP17_D1_PRESERVED_CURRENT_PATH")

    def test_06_path_provenance_preserved(self):
        source = STEP26N.read_json(STEP26N.OUTPUT / "path_source.json")
        self.assertEqual(source["source_step"], "STEP_17")
        self.assertEqual(source["path_sha256"], self.path_digest)
        self.assertFalse(source["new_path_generated"])

    def test_07_same_path_before_after(self):
        self.assertEqual(self.before_path["path_sha256"], self.after_path["path_sha256"])
        self.assertEqual(self.before_path["path_grid"], self.after_path["path_grid"])
        self.assertEqual(self.before_path["path_world"], self.after_path["path_world"])
        self.assertTrue(self.summary["same_path_before_after"])

    def test_08_path_coordinates_finite(self):
        self.assertTrue(all(math.isfinite(value) for point in self.path["path_world"] for value in point))

    def test_09_path_frame_compatible(self):
        self.assertEqual(self.before_world["collision_world"]["coordinate_frame"], "LEGACY_WORKSPACE_COORDINATE")
        self.assertEqual(self.after_world["collision_world"]["coordinate_frame"], "LEGACY_WORKSPACE_COORDINATE")

    def test_10_existing_step17_validity_helper_reused(self):
        source = inspect.getsource(STEP26N)
        self.assertIn("validate_current_path", source)
        self.assertIsNotNone(validate_current_path)

    def test_11_before_valid_handling(self):
        self.assertEqual(self.before_validity["path_state"], "VALID")
        self.assertFalse(self.before_validity["replan_required"])

    def test_12_before_blocked_handling(self):
        self.assertEqual(STEP26N.classify_path_effect("BLOCKED", "BLOCKED"), "PREEXISTING_BLOCK_OR_NONATTRIBUTABLE")

    def test_13_after_valid_handling(self):
        self.assertEqual(self.after_validity["path_state"], "VALID")
        self.assertFalse(self.after_validity["replan_required"])

    def test_14_after_blocked_handling(self):
        self.assertEqual(STEP26N.classify_path_effect("VALID", "BLOCKED"), "OBSERVATION_INDUCED_BLOCK")

    def test_15_valid_to_blocked_classification(self):
        self.assertEqual(STEP26N.classify_path_effect("VALID", "BLOCKED"), "OBSERVATION_INDUCED_BLOCK")

    def test_16_valid_to_valid_classification(self):
        self.assertEqual(STEP26N.classify_path_effect("VALID", "VALID"), "NO_PATH_EFFECT")
        self.assertEqual(self.summary["effect_classification"], "NO_PATH_EFFECT")

    def test_17_blocked_to_blocked_classification(self):
        self.assertEqual(STEP26N.classify_path_effect("BLOCKED", "BLOCKED"), "PREEXISTING_BLOCK_OR_NONATTRIBUTABLE")

    def test_18_blocked_to_valid_is_unexpected(self):
        self.assertEqual(STEP26N.classify_path_effect("BLOCKED", "VALID"), "UNEXPECTED_STATE_TRANSITION")

    def test_19_first_blocking_location_supported(self):
        audit = STEP26N.read_json(STEP26N.OUTPUT / "step17_path_contract_audit.json")
        self.assertEqual(audit["first_blocking_support"], ["waypoint_index", "edge_index", "blocking_obstacle_ids"])
        self.assertIsNone(self.provenance["first_blocked_waypoint_index"])
        self.assertIsNone(self.provenance["first_blocked_edge_index"])

    def test_20_new_collision_cell_attribution_supported(self):
        changed = next(item for item in self.occupancy["changed_cells"] if item[2:] == [0, 1])
        path = {"path_grid": [[changed[0], changed[1]]], "path_world": [[0.0, 0.0]]}
        before = {"path_state": "VALID"}
        after = {"path_state": "BLOCKED", "first_blocked_waypoint_index": 0, "first_blocked_edge_index": None, "blocking_obstacle_ids": []}
        result = STEP26N.blocking_provenance(before, after, path, self.occupancy, self.obstacle["id"])
        self.assertTrue(result["blocking_cell_newly_occupied"])
        self.assertEqual(result["dynamic_obstacle_attribution"], "SUPPORTED")

    def test_21_no_false_attribution_for_preexisting_block(self):
        path = {"path_grid": [[35, 38]], "path_world": [[0.77, 0.71]]}
        before = {"path_state": "BLOCKED"}
        after = {"path_state": "BLOCKED", "first_blocked_waypoint_index": 0, "first_blocked_edge_index": None, "blocking_obstacle_ids": [self.obstacle["id"]]}
        result = STEP26N.blocking_provenance(before, after, path, self.occupancy, self.obstacle["id"])
        self.assertEqual(result["dynamic_obstacle_attribution"], "NOT_SUPPORTED")

    def test_22_static_dynamic_provenance_preserved(self):
        before_static = [item for item in self.before_world["obstacle_state"]["obstacles"] if item["state_role"] == "STATIC_OBSTACLE"]
        after_static = [item for item in self.after_world["obstacle_state"]["obstacles"] if item["state_role"] == "STATIC_OBSTACLE"]
        self.assertEqual(before_static, after_static)
        self.assertEqual(self.obstacle["state_role"], "DYNAMIC_OBSTACLE")

    def test_23_no_physical_size_inference(self):
        text = json.dumps(self.summary) + inspect.getsource(STEP26N)
        self.assertNotIn("object_radius", text)
        self.assertNotIn("physical_footprint", text)
        self.assertEqual(self.summary["physical_geometry"], "NOT_READY_NOT_USED")

    def test_24_no_additional_inflation(self):
        self.assertEqual(self.before_world["collision_world"]["safety_margin_m"], 0.0)
        self.assertEqual(self.after_world["collision_world"]["safety_margin_m"], 0.0)

    def test_25_no_safety_margin_duplication(self):
        self.assertNotIn("safety_margin", inspect.getsource(STEP26N))
        self.assertEqual(self.occupancy["occupied_to_free_count"], 0)

    def test_26_step23m_unchanged(self):
        measurement = STEP26N.read_json(STEP26N.STEP23M)
        self.assertEqual(measurement["measurement_input_status"], "WAITING_FOR_USER_MEASUREMENT")
        self.assertEqual(measurement["representative_physical_geometry"], "NOT_READY")

    def test_27_qwen_inference_zero(self):
        self.assertEqual(self.summary["qwen_inference_runs"], 0)

    def test_28_homography_recomputation_zero(self):
        self.assertEqual(self.summary["homography_recomputation"], "NOT_RUN")

    def test_29_step18_not_called(self):
        self.assertEqual(self.summary["step18_call"], "NOT_RUN")

    def test_30_step19_not_called(self):
        self.assertEqual(self.summary["step19_call"], "NOT_RUN")

    def test_31_astar_not_called(self):
        self.assertEqual(self.summary["astar"], "NOT_RUN")
        self.assertNotIn("plan_path", inspect.getsource(STEP26N))

    def test_32_protected_prior_artifacts_unchanged(self):
        self.assertEqual(STEP26N.snapshot_protected(), STEP26N.PROTECTED_BASELINE)
        self.assertTrue(self.summary["protected_artifacts_unchanged"])


if __name__ == "__main__":
    unittest.main()
