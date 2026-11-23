#!/usr/bin/env python3
"""Regenerate the P01 zone draft overlay from zones_draft.json (CPU only)."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
SPEC = HERE / "zones_draft.json"
OUT = HERE / "zones_overlay_P01.png"
COLORS = {"yellow_area": (0, 165, 255), "green_walkway": (255, 255, 0), "other": (255, 80, 255)}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    spec = json.loads(SPEC.read_text())
    image_path, annotation_path = Path(spec["image"]), Path(spec["annotation"])
    if sha256(image_path) != spec["image_sha256"] or sha256(annotation_path) != spec["annotation_sha256"]:
        raise SystemExit("P01 image or annotation differs from the zone draft source")
    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f"cannot read image: {image_path}")
    height, width = image.shape[:2]
    cs = spec["coordinate_system"]
    if (width, height) != (cs["width"], cs["height"]):
        raise SystemExit("image size and raw-pixel coordinate system differ")

    annotation = json.loads(annotation_path.read_text())
    if annotation["image_sha256"] != spec["image_sha256"]:
        raise SystemExit("annotation refers to another image")
    for obj in annotation["objects"]:
        x1, y1, x2, y2 = (round(v) for v in obj["box_xyxy"])
        cv2.rectangle(image, (x1, y1), (x2, y2), (170, 170, 170), 1)

    for zone in spec["zones"]:
        pts = np.asarray(zone["polygon_xy"], dtype=np.int32)
        if len(pts) < 3 or np.any(pts < 0) or np.any(pts[:, 0] >= width) or np.any(pts[:, 1] >= height):
            raise SystemExit(f"invalid raw-pixel polygon: {zone['id']}")
        color = COLORS[zone["type"]]
        fill = image.copy()
        cv2.fillPoly(fill, [pts], color)
        cv2.addWeighted(fill, 0.12, image, 0.88, 0, image)
        cv2.polylines(image, [pts], True, color, 3, cv2.LINE_AA)
        for x, y in pts:
            cv2.circle(image, (int(x), int(y)), 4, color, -1, cv2.LINE_AA)
        cx, cy = zone["label_anchor_xy"]
        label = f"{zone['id']} {zone['type']}"
        cv2.rectangle(image, (cx - 8, cy - 28), (cx + 14 * len(label), cy + 8), (0, 0, 0), -1)
        cv2.putText(image, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)

    cv2.rectangle(image, (15, 15), (590, 82), (0, 0, 0), -1)
    cv2.putText(image, "P01 RAW 1920x1080  |  DRAFT ZONES", (28, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, "thin gray = answer-key object boxes", (28, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (185, 185, 185), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(OUT), image):
        raise SystemExit(f"cannot write overlay: {OUT}")
    print(OUT)


if __name__ == "__main__":
    main()
