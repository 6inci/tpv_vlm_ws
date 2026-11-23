import importlib.util
from pathlib import Path
import unittest


WORKSPACE = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("step30n", WORKSPACE / "scripts" / "audit_offline_navigation_chain.py")
STEP30N = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STEP30N)


class OfflineNavigationChainIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for name in (
            "observed_track_lineage", "development_track_lineage", "cross_step_consistency",
            "evidence_class_separation", "system_capability_matrix", "system_limitations", "step30n_summary",
        ):
            setattr(cls, name, STEP30N.read_json(STEP30N.OUTPUT / f"{name}.json"))

    def test_01_step22d_exists_and_passes(self):
        self.assertEqual(STEP30N.read_json(STEP30N.STEP22D / "step22d_summary.json")["step_result"], "PASS")

    def test_02_step24n_links_step22d(self):
        self.assertEqual(self.observed_track_lineage["source"]["image_id"], "image_014_photo_08")
        self.assertEqual(self.observed_track_lineage["source"]["track_id"], "track_039")

    def test_03_step25n_identity_links_step24n(self):
        self.assertEqual(self.observed_track_lineage["navigation"]["dynamic_obstacle_id"], "dynamic_image_014_photo_08_track_039")

    def test_04_step26n_observed_path_valid(self):
        self.assertEqual(self.observed_track_lineage["navigation"]["path_validity"], "VALID")
        self.assertEqual(self.observed_track_lineage["navigation"]["path_effect"], "NO_PATH_EFFECT")

    def test_05_case_a_provenance(self):
        self.assertEqual(self.observed_track_lineage["evidence_class"], "OBSERVED_NAVIGATION_APPROXIMATION")

    def test_06_case_a_step18_zero(self):
        self.assertEqual(self.observed_track_lineage["step18_invocations"], 0)

    def test_07_case_a_no_replacement(self):
        self.assertEqual(self.observed_track_lineage["replacement_path"], "NOT_GENERATED")

    def test_08_step27n_fixture_identity(self):
        self.assertEqual(self.development_track_lineage["fixture"]["scenario_id"], "D2")
        self.assertEqual(self.development_track_lineage["fixture"]["event_id"], "D2_add_on_path")

    def test_09_dynamic_box_preserved(self):
        self.assertEqual(self.development_track_lineage["fixture"]["obstacle_id"], "dynamic_box_02")

    def test_10_step28n_persistent(self):
        self.assertEqual(self.development_track_lineage["persistence"]["classification"], "PERSISTENT_BLOCKING")

    def test_11_step28n_replan_required(self):
        self.assertEqual(self.development_track_lineage["persistence"]["decision"], "REPLAN_REQUIRED")

    def test_12_step29n_invocation_one(self):
        self.assertEqual(self.development_track_lineage["replanning"]["invocations"], 1)

    def test_13_step29n_replan_success(self):
        self.assertEqual(self.development_track_lineage["replanning"]["result"], "REPLAN_SUCCESS")

    def test_14_replacement_valid(self):
        self.assertEqual(self.development_track_lineage["replacement_path"]["validation"], "VALID")

    def test_15_revision_zero_to_one(self):
        self.assertEqual(self.development_track_lineage["old_path"]["revision"], 0)
        self.assertEqual(self.development_track_lineage["replacement_path"]["revision"], 1)

    def test_16_evidence_classes_not_merged(self):
        self.assertFalse(self.evidence_class_separation["tracks_combined"])
        self.assertNotEqual(self.observed_track_lineage["evidence_class"], self.development_track_lineage["evidence_class"])

    def test_17_fixture_not_camera_derived(self):
        self.assertFalse(self.development_track_lineage["camera_vlm_derived"])

    def test_18_observed_case_not_blocking(self):
        self.assertFalse(self.evidence_class_separation["observed_track"]["blocking_observed"])
        self.assertEqual(self.observed_track_lineage["policy"], "CONTINUE_CURRENT_PATH")

    def test_19_step23m_unchanged(self):
        self.assertEqual(STEP30N.read_json(STEP30N.STEP23M)["measurement_input_status"], "WAITING_FOR_USER_MEASUREMENT")

    def test_20_no_physical_size_inference(self):
        self.assertEqual(self.observed_track_lineage["physical_geometry"], "NOT_READY_NOT_USED")
        self.assertEqual(self.step30n_summary["physical_geometry"], "NOT_READY_NOT_INFERRED")

    def test_21_no_qwen(self):
        self.assertEqual(self.step30n_summary["qwen_inference_runs"], 0)

    def test_22_no_homography(self):
        self.assertEqual(self.step30n_summary["homography_recomputation"], "NOT_RUN")

    def test_23_no_ros_nav2(self):
        self.assertEqual(self.step30n_summary["ros_nav2"], "NOT_RUN")
        self.assertEqual(self.step30n_summary["hardware_execution"], "NOT_RUN")

    def test_24_prior_artifacts_unchanged(self):
        self.assertTrue(self.step30n_summary["protected_artifacts_unchanged"])
        self.assertEqual(STEP30N.snapshot_protected(), self.step30n_summary["protected_artifacts_after"])


if __name__ == "__main__":
    unittest.main()
