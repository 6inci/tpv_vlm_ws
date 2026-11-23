import copy
import json
from pathlib import Path
import sys
import unittest


WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))

from vlm_reposition.world import (
    StateValidationError,
    build_state_from_saved_results,
    build_workspace_task_state,
    get_zone,
    load_zones,
)


TASK_DIR = WORKSPACE / "results" / "task_parser_rework"
TARGET_DIR = WORKSPACE / "results" / "task_conditioned_grounding"
SCENE_PATH = WORKSPACE / "results" / "semantic_refinement" / "semantic_run01.json"
ZONES_PATH = WORKSPACE / "config" / "zones.json"


class WorkspaceStateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.zones = load_zones(ZONES_PATH)
        cls.scene = json.loads(SCENE_PATH.read_text())["entities"]
        task_record = json.loads((TASK_DIR / "valid_ko_red_cup_zone_a_run01.json").read_text())
        cls.task = task_record["validated"]
        cls.target = json.loads((TARGET_DIR / "red_cup_run01.json").read_text())
        cls.zone = get_zone(cls.zones, "zone_A")

    def build(self, task=None, target=None, zone=None, scene=None, zones=None):
        return build_workspace_task_state(
            self.task if task is None else task,
            self.target if target is None else target,
            self.zone if zone is None else zone,
            self.scene if scene is None else scene,
            self.zones if zones is None else zones,
        )

    def test_valid_state_build(self):
        state = self.build()
        self.assertEqual(state["state_validity"], "VALID_REPRESENTATION")
        self.assertEqual(state["target"]["grounding_status"], "TARGET_FOUND")
        self.assertEqual(state["destination"]["zone_id"], "zone_A")
        self.assertEqual(state["scene"]["entity_count"], 11)
        self.assertEqual(state["scene"]["entities"][6]["geometry_status"], "VALID")
        self.assertEqual(state["scene"]["entities"][6]["reference_review_status"], "MATCHED_REFERENCE")

    def test_invalid_task_rejected(self):
        cases = (
            ("unsupported_action_run01.json", "UNSUPPORTED_ACTION"),
            ("missing_destination_run01.json", "TASK_INCOMPLETE"),
            ("ambiguous_target_run01.json", "TARGET_AMBIGUOUS"),
        )
        for filename, status in cases:
            task = json.loads((TASK_DIR / filename).read_text())["validated"]
            with self.subTest(status=status), self.assertRaises(StateValidationError) as context:
                self.build(task=task)
            self.assertEqual(context.exception.code, "INVALID_TASK")

    def test_target_not_found_rejected(self):
        target = copy.deepcopy(self.target)
        target["target_status"] = "TARGET_NOT_FOUND"
        with self.assertRaises(StateValidationError) as context:
            self.build(target=target)
        self.assertEqual(context.exception.code, "TARGET_NOT_FOUND")

    def test_zone_missing_rejected(self):
        task = copy.deepcopy(self.task)
        task["destination"]["zone_id"] = "zone_missing"
        with self.assertRaises(StateValidationError) as context:
            self.build(task=task)
        self.assertEqual(context.exception.code, "DESTINATION_NOT_RESOLVED")

    def test_frame_mismatch_rejected(self):
        target = copy.deepcopy(self.target)
        target["world_coordinate"]["frame"] = "OTHER_FRAME"
        with self.assertRaises(StateValidationError) as context:
            self.build(target=target)
        self.assertEqual(context.exception.code, "FRAME_MISMATCH")

    def test_non_finite_coordinate_rejected(self):
        target = copy.deepcopy(self.target)
        target["world_coordinate"]["x"] = float("nan")
        with self.assertRaises(StateValidationError) as context:
            self.build(target=target)
        self.assertEqual(context.exception.code, "NON_FINITE_COORDINATE")

    def test_outside_target_and_optional_scene_association(self):
        task = json.loads((TASK_DIR / "valid_ko_white_bag_zone_b_run01.json").read_text())["validated"]
        target = json.loads((TARGET_DIR / "white_bag_run01.json").read_text())
        state = self.build(task=task, target=target, zone=get_zone(self.zones, "zone_B"))
        self.assertEqual(state["state_validity"], "VALID_REPRESENTATION")
        self.assertEqual(state["target"]["active_workspace_status"], "OUTSIDE")
        self.assertIsNone(state["target"]["scene_association"]["scene_entity_id"])

    def test_target_scene_iou_association(self):
        state = self.build()
        association = state["target"]["scene_association"]
        self.assertEqual(association["status"], "ASSOCIATED_BY_BBOX_IOU")
        self.assertEqual(association["scene_entity_id"], "entity_007")
        self.assertGreaterEqual(association["bbox_iou"], association["threshold"])

    def test_deferred_fields_remain_explicit(self):
        state = self.build()
        self.assertEqual(state["feasibility"]["status"], "NOT_EVALUATED")
        self.assertEqual(state["planning"]["status"], "NOT_RUN")
        self.assertEqual(state["robot"], {"status": "NOT_AVAILABLE"})
        self.assertNotIn("robot_goal_pose", state)
        self.assertNotIn("path", state["planning"])

    def test_saved_result_integration_and_determinism(self):
        cases = (
            ("valid_ko_red_cup_zone_a_run01.json", "red_cup_run01.json", "zone_A", "INSIDE"),
            ("valid_ko_white_bag_zone_b_run01.json", "white_bag_run01.json", "zone_B", "OUTSIDE"),
            ("valid_ko_trophy_zone_f_run01.json", "gold_trophy_like_object_run01.json", "zone_F", "INSIDE"),
        )
        for task_name, target_name, zone_id, workspace_status in cases:
            with self.subTest(target=target_name):
                first = build_state_from_saved_results(
                    TASK_DIR / task_name, TARGET_DIR / target_name, SCENE_PATH, ZONES_PATH
                )
                second = build_state_from_saved_results(
                    TASK_DIR / task_name, TARGET_DIR / target_name, SCENE_PATH, ZONES_PATH
                )
                self.assertEqual(first, second)
                self.assertEqual(first["destination"]["zone_id"], zone_id)
                self.assertEqual(first["target"]["active_workspace_status"], workspace_status)
                self.assertEqual(first["provenance"]["source_statuses"]["scene"], "PARTIAL")


if __name__ == "__main__":
    unittest.main()
