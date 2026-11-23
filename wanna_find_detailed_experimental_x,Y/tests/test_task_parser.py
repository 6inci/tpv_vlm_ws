import json
from pathlib import Path
import sys
import unittest


WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))

from vlm_reposition.task import (
    TaskValidationError,
    assemble_integrated_task,
    canonicalize_target_phrase,
    evaluate_task_output,
    load_task_language,
    parse_task_model_output,
    resolve_action_phrase,
    validate_task_fields,
)
from vlm_reposition.world import load_zones


CONFIG = json.loads((WORKSPACE / "config" / "task_parser.json").read_text())


class TaskParserTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.zones = load_zones(CONFIG["zone_config_path"])
        cls.language = load_task_language(CONFIG["task_language_config_path"])

    def validate(self, output, instruction="빨간 컵을 A구역으로 옮겨"):
        parsed = parse_task_model_output(json.dumps(output, ensure_ascii=False))
        return validate_task_fields(instruction, parsed, self.zones, self.language)

    def test_language_config_load(self):
        self.assertIn("reposition", self.language["supported_actions"])
        self.assertEqual(len(self.language["target_aliases"]), 3)

    def test_valid_schema_and_zone_resolution(self):
        result = self.validate({
            "action_phrase": "place",
            "target_phrase": "the gold trophy",
            "destination_phrase": "F 구역",
        }, "Place the gold trophy in Zone F.")
        self.assertEqual(result["status"], "VALID_TASK")
        self.assertEqual(result["action"]["canonical"], "reposition")
        self.assertEqual(result["target"]["grounding_phrase"], "gold trophy-like object")
        self.assertEqual(result["destination"]["zone_id"], "zone_F")

    def test_json_parse_error(self):
        with self.assertRaisesRegex(TaskValidationError, "no JSON object") as context:
            parse_task_model_output("not json")
        self.assertEqual(context.exception.code, "JSON_PARSE_ERROR")

    def test_missing_and_invalid_fields(self):
        with self.assertRaises(TaskValidationError) as context:
            parse_task_model_output('{"action_phrase":"move"}')
        self.assertIn(context.exception.code, {"MISSING_TARGET", "MISSING_DESTINATION"})
        with self.assertRaises(TaskValidationError) as context:
            parse_task_model_output(
                '{"action_phrase":"move","target_phrase":[],"destination_phrase":"A"}'
            )
        self.assertEqual(context.exception.code, "INVALID_SCHEMA")

    def test_missing_target_and_destination(self):
        missing_target = self.validate({
            "action_phrase": "옮겨",
            "target_phrase": None,
            "destination_phrase": "A구역",
        }, "A구역으로 옮겨")
        missing_destination = self.validate({
            "action_phrase": "옮겨",
            "target_phrase": "빨간 컵",
            "destination_phrase": "null",
        }, "빨간 컵을 옮겨")
        self.assertEqual(missing_target["error_code"], "MISSING_TARGET")
        self.assertEqual(missing_destination["error_code"], "MISSING_DESTINATION")

    def test_action_resolution_table(self):
        for phrase in ("옮겨", "이동시켜", "가져다 놔", "가져다 놓아", "move", "place"):
            result = resolve_action_phrase(phrase, self.language)
            self.assertEqual(result["canonical"], "reposition")
            self.assertEqual(result["resolution_status"], "SUPPORTED_ACTION")
        for phrase in ("회전시켜", "rotate"):
            result = resolve_action_phrase(phrase, self.language)
            self.assertEqual(result["resolution_status"], "UNSUPPORTED_ACTION")

    def test_raw_instruction_guard_overrides_model_and_bad_json(self):
        hallucinated = '{"action_phrase":"옮겨","target_phrase":"빨간 컵","destination_phrase":null}'
        _, result = evaluate_task_output(
            "빨간 컵을 회전시켜", hallucinated, self.zones, self.language
        )
        self.assertEqual(result["status"], "UNSUPPORTED_ACTION")
        self.assertEqual(result["action"]["resolution_status"], "UNSUPPORTED_ACTION_RAW_GUARD")
        self.assertEqual(result["target_grounding_status"], "NOT_RUN")
        _, malformed_result = evaluate_task_output(
            "Rotate the red cup.", "not json", self.zones, self.language
        )
        self.assertEqual(malformed_result["status"], "UNSUPPORTED_ACTION")

    def test_unknown_zone_is_not_resolved(self):
        result = self.validate({
            "action_phrase": "옮겨",
            "target_phrase": "빨간 컵",
            "destination_phrase": "왼쪽",
        }, "빨간 컵을 왼쪽으로 옮겨")
        self.assertEqual(result["status"], "DESTINATION_NOT_RESOLVED")

    def test_target_canonicalization_table(self):
        expected = {
            "빨간 컵": "red cup",
            "흰색 가방": "white bag",
            "금색 트로피": "gold trophy-like object",
            "red cup": "red cup",
        }
        for surface, grounding in expected.items():
            result = canonicalize_target_phrase(surface, self.language)
            self.assertEqual(result["grounding_phrase"], grounding)
            self.assertEqual(result["canonicalization_status"], "KNOWN_ALIAS")

    def test_open_vocabulary_target_is_preserved(self):
        result = canonicalize_target_phrase("파란 상자", self.language)
        self.assertEqual(result, {
            "surface_phrase": "파란 상자",
            "grounding_phrase": "파란 상자",
            "canonicalization_status": "OPEN_VOCABULARY",
        })

    def test_referential_target_is_ambiguous(self):
        target = canonicalize_target_phrase("저걸", self.language)
        self.assertEqual(target["canonicalization_status"], "TARGET_AMBIGUOUS")
        result = self.validate({
            "action_phrase": "옮겨",
            "target_phrase": "저걸",
            "destination_phrase": "A구역",
        }, "저걸 A구역으로 옮겨")
        self.assertEqual(result["status"], "TARGET_AMBIGUOUS")

    def test_structured_task_preserves_workspace_status(self):
        task = self.validate({
            "action_phrase": "이동시켜",
            "target_phrase": "흰색 가방",
            "destination_phrase": "B구역",
        }, "흰색 가방을 B구역으로 이동시켜")
        grounding = json.loads(
            (Path(CONFIG["target_grounding_result_dir"]) / "white_bag_run01.json").read_text()
        )
        integrated = assemble_integrated_task(task, self.zones, grounding)
        self.assertEqual(integrated["target"]["grounding_phrase"], "white bag")
        self.assertEqual(integrated["target"]["grounding"]["active_workspace_status"], "OUTSIDE")
        self.assertEqual(integrated["destination"]["zone_id"], "zone_B")
        self.assertEqual(integrated["feasibility_status"], "NOT_EVALUATED")
        self.assertEqual(integrated["path_status"], "NOT_RUN")


if __name__ == "__main__":
    unittest.main()
