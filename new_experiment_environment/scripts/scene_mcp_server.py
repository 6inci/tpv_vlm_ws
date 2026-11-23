#!/usr/bin/env python3
"""Minimal read-only JSON-RPC/MCP stdio prototype for one generated scene."""

import argparse
import json
from pathlib import Path
import sys


TOOLS = [
    {"name": "query_scene_graph", "description": "Find scene nodes by label/category.",
     "inputSchema": {"type": "object", "properties": {"label": {"type": "string"}}, "required": ["label"]}},
    {"name": "get_object_metric_position", "description": "Return one object node and its representative metric XY.",
     "inputSchema": {"type": "object", "properties": {"object_id": {"type": "string"}}, "required": ["object_id"]}},
    {"name": "get_semantic_regions", "description": "Return semantic-region records.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "get_metric_map", "description": "Return metric-map metadata and artifact paths.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "get_traversability_map", "description": "Return bbox/SAM traversability artifact paths.",
     "inputSchema": {"type": "object", "properties": {}}},
]


def read_json(path):
    return json.loads(Path(path).read_text())


def call_tool(scene_path, name, arguments):
    scene_path = Path(scene_path).resolve()
    scene = read_json(scene_path)
    run = scene_path.parents[1]
    if name == "query_scene_graph":
        query = arguments.get("label", "").casefold()
        return [node for node in scene["nodes"] if query in (node.get("semantic_label") or node.get("label") or "").casefold() or query == (node.get("canonical_category") or "").casefold()]
    if name == "get_object_metric_position":
        object_id = arguments.get("object_id")
        return next((node for node in scene["nodes"] if node.get("id") == object_id), None)
    if name == "get_semantic_regions":
        return scene.get("semantic_regions", [])
    if name == "get_metric_map":
        return read_json(run / "03_metric_map/metric_map.json")
    if name == "get_traversability_map":
        data = read_json(run / "03_metric_map/metric_map.json")
        return {"status": data["status"], "bbox": data["artifacts"]["traversability_bbox"], "sam2": data["artifacts"].get("traversability_sam2")}
    raise ValueError(f"unknown tool: {name}")


def response(message, scene_path):
    method = message.get("method")
    request_id = message.get("id")
    if method == "initialize":
        result = {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}}, "serverInfo": {"name": "synthetic-scene-reader", "version": "0.1"}}
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "tools/call":
        params = message.get("params", {})
        value = call_tool(scene_path, params.get("name"), params.get("arguments", {}))
        result = {"content": [{"type": "text", "text": json.dumps(value, ensure_ascii=False)}], "isError": False}
    else:
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32601, "message": "Method not found"}}
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path)
    parser.add_argument("--once", choices=[tool["name"] for tool in TOOLS])
    parser.add_argument("--args", default="{}")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        assert len({tool["name"] for tool in TOOLS}) == len(TOOLS)
        assert response({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, Path("unused"))["result"]["tools"] == TOOLS
        print("self-check: PASS")
        return
    if not args.scene:
        parser.error("--scene is required")
    if args.once:
        print(json.dumps(call_tool(args.scene, args.once, json.loads(args.args)), indent=2, ensure_ascii=False))
        return
    for line in sys.stdin:
        try:
            print(json.dumps(response(json.loads(line), args.scene), ensure_ascii=False), flush=True)
        except Exception as exc:
            print(json.dumps({"jsonrpc": "2.0", "id": None, "error": {"code": -32000, "message": str(exc)}}), flush=True)


if __name__ == "__main__":
    main()
