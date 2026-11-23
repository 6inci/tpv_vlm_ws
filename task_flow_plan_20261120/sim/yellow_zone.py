#!/usr/bin/env python3
"""CPU probe: can the drivable zone be derived from the yellow floor lines without a hand-drawn map?

Method: yellow tape (HSV) = barrier; drivable zone = floor region connected to the robot position
without crossing a barrier. Compared with the hand-drawn DRAFT zones (Z02 minus Z03/Z04), which are
themselves NOT human-confirmed, so the IoU is an agreement figure, not an accuracy.
Reads photos and the DRAFT map only. Writes under sim/results/<stamp>_yellow_zone/.
Coordinates: undistorted_px.
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/home/binci/vlm_ws/task_flow_plan_20261120")
PHOTO_DIR = Path("/home/binci/Pictures/Calibration")
PHOTOS = {1: "1calibration.png", 2: "2calibration.png", 3: "3calibration.png", 4: "4calibration.png",
          5: "5calibration.png", 6: "6calibration.png", 7: "7calibration.png",
          8: "undistorted_26007129092710.png"}
DRAFT = Path("/home/binci/vlm_ws/semantic_task_navigation/experiments/first_vlm_navigation_path/results/"
             "20261119_001504_first_path/map_E1_transferred.json")
ROBOT_SEED = (1709, 697)          # photo_5 wheel centre (navigation_config.yaml); robot is there in all 8 photos
HSV_LO, HSV_HI = (18, 60, 120), (40, 255, 255)   # yellow tape, declared before the first run
CLOSE_PX = 9                      # closes striped hazard tape and seam gaps
# Run 2 (after run 1 leaked: tape saturation is only 29-127 and the right border is striped tape, not yellow):
# barrier = anything bright on the dark mat. Usage: yellow_zone.py bright
METHOD = sys.argv[1] if len(sys.argv) > 1 else "yellow"
BRIGHT_V, BRIGHT_CLOSE_PX = 130, 15


def draft_mask(shape):
    d = json.loads(DRAFT.read_text())
    m = np.zeros(shape, np.uint8)
    for z in d["traversable"]:
        cv2.fillPoly(m, [np.round(np.array(z["polygon"])).astype(np.int32)], 1)
    for z in d["static_obstacles"]:
        cv2.fillPoly(m, [np.round(np.array(z["polygon"])).astype(np.int32)], 0)
    return m


def main():
    out = ROOT / "sim" / "results" / (time.strftime("%Y%m%d_%H%M%S") + f"_zone_{METHOD}")
    out.mkdir(parents=True)
    rows = []
    for p, name in PHOTOS.items():
        img = cv2.imread(str(PHOTO_DIR / name))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        if METHOD == "yellow":
            yellow = cv2.inRange(hsv, HSV_LO, HSV_HI)
            barrier = cv2.morphologyEx(yellow, cv2.MORPH_CLOSE, np.ones((CLOSE_PX, CLOSE_PX), np.uint8))
        else:
            bright = (hsv[..., 2] > BRIGHT_V).astype(np.uint8) * 255
            barrier = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((BRIGHT_CLOSE_PX, BRIGHT_CLOSE_PX), np.uint8))
        free = (barrier == 0).astype(np.uint8)
        n, lab = cv2.connectedComponents(free, connectivity=4)
        # Run 3 fix: in photos 2-3 the seed pixel itself was a bright robot part (label 0 = barrier).
        # Seed = the most common free-floor label in a 240 px window around the robot.
        win = lab[ROBOT_SEED[1] - 120:ROBOT_SEED[1] + 120, ROBOT_SEED[0] - 120:ROBOT_SEED[0] + 120]
        ids, cnt = np.unique(win[win > 0], return_counts=True)
        zone = (lab == ids[np.argmax(cnt)]).astype(np.uint8)
        ref = draft_mask(zone.shape)
        inter, union = int((zone & ref).sum()), int((zone | ref).sum())
        leaked = bool(zone[5, 5])            # zone reaches the image corner = the line loop is not closed
        rows.append({"photo": p, "zone_px": int(zone.sum()), "draft_px": int(ref.sum()),
                     "iou_vs_draft": round(inter / union, 4), "zone_frac_of_image": round(float(zone.mean()), 4),
                     "leaked_to_image_corner": leaked})
        vis = img.copy()
        vis[zone == 1] = (0.55 * vis[zone == 1] + 0.45 * np.array([0, 200, 0])).astype(np.uint8)
        cv2.polylines(vis, [np.round(np.array(json.loads(DRAFT.read_text())["traversable"][0]["polygon"])).astype(np.int32)],
                      True, (255, 0, 255), 2)
        cv2.imwrite(str(out / f"photo{p}_zone.jpg"), vis)
        cv2.imwrite(str(out / f"photo{p}_yellow.png"), barrier)
        print(rows[-1])
    (out / "summary.json").write_text(json.dumps({"method": METHOD, "hsv": [HSV_LO, HSV_HI], "close_px": CLOSE_PX, "bright_v": BRIGHT_V, "bright_close_px": BRIGHT_CLOSE_PX, "seed": ROBOT_SEED,
                                                  "rows": rows}, indent=1))
    print(out)


if __name__ == "__main__":
    sys.exit(main())
