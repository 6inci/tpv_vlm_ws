#!/usr/bin/env python3
"""Audit stored research images against the photo_04 legacy geometry reference."""

import argparse
import json
from pathlib import Path
import sys


WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))

from vlm_reposition.perception.image_geometry_audit import (
    audit_geometry,
    audit_legacy_geometry,
    build_replay_manifest,
    inventory_images,
    load_audit_config,
    write_registration_visualizations,
)


DEFAULT_CONFIG = WORKSPACE / "config" / "offline_image_geometry_audit.json"
DEFAULT_MANIFEST = WORKSPACE / "config" / "offline_real_image_replay.json"
DEFAULT_OUTPUT = WORKSPACE / "results" / "offline_image_replay"


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest-output", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output_dir.exists() or args.manifest_output.exists():
        raise FileExistsError("refusing to overwrite existing audit output or replay manifest")

    config = load_audit_config(args.config)
    inventory = inventory_images(config["candidate_directory"], config["reference_image"])
    audit, diagnostics = audit_geometry(config, inventory, include_diagnostics=True)
    repeated, _ = audit_geometry(config, inventory)
    deterministic = audit == repeated
    if not deterministic:
        raise RuntimeError("geometry audit was not deterministic")
    legacy = audit_legacy_geometry(config)
    manifest = build_replay_manifest(config, audit)

    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir / "image_inventory.json", inventory)
    write_json(args.output_dir / "registration_metrics.json", {
        "schema_version": "0.1",
        "diagnostic_transform_role": "IMAGE_TO_IMAGE_COMPATIBILITY_DIAGNOSTIC_ONLY_NOT_METRIC_HOMOGRAPHY",
        "images": audit["images"],
    })
    write_json(args.output_dir / "geometry_compatibility_audit.json", {
        **audit,
        "legacy_geometry_audit": legacy,
        "deterministic_repeat": deterministic,
    })
    write_json(args.output_dir / "offline_replay_manifest.json", manifest)
    write_json(args.manifest_output, manifest)
    write_registration_visualizations(
        args.output_dir / "registration_visualizations", config["reference_image"], audit, diagnostics
    )
    print(json.dumps({
        "software_result": "PASS",
        "image_count": inventory["image_count"],
        "compatible_candidate_count": audit["compatible_candidate_count"],
        "dataset_usability_status": audit["dataset_usability_status"],
        "deterministic_repeat": deterministic,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
