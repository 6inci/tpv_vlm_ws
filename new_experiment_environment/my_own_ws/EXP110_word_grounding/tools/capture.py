#!/usr/bin/env python3
"""Capture raw camera frames for one layout (EXP110).

Camera open/size logic ported from /home/binci/camera_calibration_1920x1080/compare_raw_undistorted.py (read-only reference).

Usage (qwen_vl env):
  python tools/capture.py --split dev --layout L01 --note "robot near truck, pallets swapped"
    -> live preview; press  c = capture frames_per_layout frames,  q/ESC = quit
  python tools/capture.py --split pilot --layout T00 --auto      (no preview, capture immediately)
Saves lossless PNG (raw, distorted pixels) + one JSON metadata file per frame. Never overwrites.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / 'config.json').read_text())
CAM = CFG['camera']


def v4l2_controls(device):
    """Snapshot of camera controls (name -> value) at capture time; empty dict if v4l2-ctl is unavailable."""
    try:
        out = subprocess.run(['v4l2-ctl', '-d', device, '--list-ctrls'], capture_output=True, text=True, timeout=5).stdout
    except FileNotFoundError:
        return {}
    ctrls = {}
    for line in out.splitlines():
        if 'value=' in line:
            name = line.split()[0]
            ctrls[name] = line.split('value=')[1].split()[0]
    return ctrls


def device_index(device):
    """Stable by-id path -> current /dev/videoN index (this OpenCV build cannot open V4L2 devices by path)."""
    real = Path(device).resolve()
    if not real.name.startswith('video'):
        raise SystemExit(f'{device} -> {real} is not a /dev/videoN device (camera unplugged?)')
    return int(real.name[len('video'):])


def open_camera():
    cam = cv2.VideoCapture(device_index(CAM['device']), cv2.CAP_V4L2)
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*CAM['fourcc']))
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAM['width'])
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM['height'])
    if not cam.isOpened():
        raise SystemExit(f"camera {CAM['device']} could not be opened")
    fourcc = int(cam.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, 'little').decode(errors='replace')
    if fourcc != CAM['fourcc']:
        raise SystemExit(f"camera negotiated {fourcc!r}, config requires {CAM['fourcc']!r}")
    for _ in range(CAM['warmup_frames']):  # let auto white balance settle
        cam.read()
    return cam, fourcc


def read_frame(cam):
    ok, frame = cam.read()
    if not ok or frame.shape[1::-1] != (CAM['width'], CAM['height']):
        raise SystemExit(f"camera must return {CAM['width']}x{CAM['height']}, got {None if not ok else frame.shape[1::-1]}")
    return frame


def capture_burst(cam, fourcc, out_dir, args):
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for i in range(1, CAM['frames_per_layout'] + 1):
        png = out_dir / f'{args.layout}_f{i:02d}.png'
        if png.exists():
            raise SystemExit(f'{png} exists - refusing to overwrite (use a new layout id)')
        frame = read_frame(cam)
        t = datetime.now(timezone.utc).astimezone().isoformat()
        if not cv2.imwrite(str(png), frame):  # PNG = lossless
            raise SystemExit(f'failed to write {png}')
        meta = {'experiment_id': CFG['experiment_id'], 'split': args.split, 'layout': args.layout, 'frame_index': i,
                'timestamp': t, 'note': args.note, 'device': CAM['device'], 'fourcc': fourcc,
                'size': [CAM['width'], CAM['height']], 'image_space': CFG['image_space'],
                'v4l2_controls': v4l2_controls(CAM['device']), 'sha256': hashlib.sha256(png.read_bytes()).hexdigest()}
        png.with_suffix('.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        saved.append(png)
        time.sleep(CAM['frame_interval_s'])
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', required=True, choices=['pilot', 'dev', 'test'])
    ap.add_argument('--layout', required=True, help='layout id, e.g. L01 (new id per arrangement)')
    ap.add_argument('--note', default='', help='what was moved / where the robot is')
    ap.add_argument('--auto', action='store_true', help='capture immediately without preview')
    ap.add_argument('--out', default=None, help='override output dir (for tests)')
    args = ap.parse_args()
    out_dir = Path(args.out) if args.out else ROOT / CFG['paths']['data'] / args.split
    cam, fourcc = open_camera()
    try:
        if args.auto:
            saved = capture_burst(cam, fourcc, out_dir, args)
        else:
            saved = []
            s = CAM['preview_scale']
            print('preview: c = capture, q/ESC = quit')
            while True:
                frame = read_frame(cam)
                cv2.imshow('EXP110 capture (preview only)', cv2.resize(frame, None, fx=s, fy=s))
                key = cv2.waitKey(1) & 0xFF
                if key == ord('c'):
                    saved = capture_burst(cam, fourcc, out_dir, args)
                    break
                if key in (ord('q'), 27):
                    break
    finally:
        cam.release()
        cv2.destroyAllWindows()
    for p in saved:
        print('saved', p)
    if not saved:
        print('nothing captured')


if __name__ == '__main__':
    sys.exit(main())
