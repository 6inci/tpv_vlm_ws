"""Test D: real Qwen3-VL / Grounding DINO through the worker (qwen_vl env, GPU).
Run with:  STNAV_RUN_MODEL_TESTS=1 python -m pytest tests/test_models_integration.py"""
import pytest

from conftest import EXAMPLE_COMMAND, REF_IMAGE
from semantic_task_nav.agent.task_parser import parse_with_llm
from semantic_task_nav.perception.grounding import dino_candidates
from semantic_task_nav.perception.model_client import ModelWorkerClient

pytestmark = pytest.mark.models


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    from semantic_task_nav.config import load_config
    c = ModelWorkerClient(load_config()["models"], tmp_path_factory.mktemp("logs"))
    c.start()
    yield c
    c.close()


def test_worker_env(client):
    assert client.env_info["cuda_available"] and client.env_info["python"].endswith("qwen_vl/bin/python")


def test_dino_produces_candidate_boxes(client, cfg):
    cands, raw = dino_candidates(client, cfg["models"], REF_IMAGE, (1920, 1080), ["truck"])
    assert cands and all(len(c["bbox_xyxy"]) == 4 for c in cands)
    assert client.loaded["dino"]["commit_hash"]


def test_qwen_parses_example_command(client, cfg):
    def gen(prompt):
        client.ensure_loaded("qwen")
        return client.call("qwen_generate", prompt=prompt, max_new_tokens=384)["text"]
    task = parse_with_llm(EXAMPLE_COMMAND, gen, cfg["models"]["prompts"]["task_parse"], "qwen3_vl_local", "qwen")
    assert task["task_type"] == "transport_and_place"
    assert task["target"]["color"] == "red" and task["destination"]["reference_object"] == "truck"
