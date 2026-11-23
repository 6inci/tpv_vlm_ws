#!/usr/bin/env python3
"""Capture lossless experiment frames from the fixed camera.

Camera open / undistortion logic ported from
/home/binci/camera_calibration_1920x1080/compare_raw_undistorted.py (read-only reference).

Usage:
  python capture_frames.py --session 2026-11-15_dev --split dev --layout L01 [--device /dev/v4l/by-id/...] [--note "..."]
  python capture_frames.py --probe --device ...   # open, read one frame, print settings, save nothing
  python capture_frames.py --selftest             # offline check, no camera
Keys in the preview: SPACE = save frame, q / ESC = quit.

Saved per frame (never overwritten):
  <out_root>/<session>/<split>_<layout>_fNN_raw.png          raw camera pixels (BGR, lossless)
  <out_root>/<session>/<split>_<layout>_fNN_undist.png       undistorted copy (only if calibration_json set; status recorded)
  <out_root>/<session>/<split>_<layout>_fNN.json             metadata
  <out_root>/<session>/manifest.jsonl                        one line per frame with sha256 (used to seal test frames)
"""
import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

CFG_PATH = Path(__file__).resolve().parent.parent / 'config' / 'capture_config.json'
PROPS = ['FRAME_WIDTH', 'FRAME_HEIGHT', 'FPS', 'AUTO_EXPOSURE', 'EXPOSURE', 'GAIN', 'BRIGHTNESS', 'CONTRAST',
         'SATURATION', 'AUTO_WB', 'WB_TEMPERATURE', 'AUTOFOCUS', 'FOCUS', 'ZOOM']


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def load_calibration(path, width, height):
    if not path:
        return None
    data = json.loads(Path(path).read_text())
    if (data['image_width'], data['image_height']) != (width, height):
        raise SystemExit(f'calibration size {data["image_width"]}x{data["image_height"]} != capture {width}x{height}')
    k = np.array(data['camera_matrix'], dtype=np.float64)
    d = np.array(data['distortion_coefficients_k1_k2_p1_p2_k3'], dtype=np.float64)
    maps = cv2.initUndistortRectifyMap(k, d, None, k, (width, height), cv2.CV_16SC2)
    return {'path': str(path), 'sha256': sha256_bytes(Path(path).read_bytes()), 'maps': maps}


def v4l2_snapshot(device):
    try:
        out = subprocess.run(['v4l2-ctl', '-d', str(device), '--all'], capture_output=True, text=True, timeout=5)
        return out.stdout if out.returncode == 0 else f'v4l2-ctl failed: {out.stderr.strip()}'
    except FileNotFoundError:
        return 'v4l2-ctl not installed'


def apply_v4l2_controls(device, controls):
    for name, value in controls.items():
        r = subprocess.run(['v4l2-ctl', '-d', str(device), '-c', f'{name}={value}'], capture_output=True, text=True)
        if r.returncode != 0:  # do not hide: a failed lock means settings are not what the config claims
            raise SystemExit(f'v4l2 control {name}={value} failed: {r.stderr.strip()}')


def open_camera(cfg, device):
    source = int(device) if str(device).isdigit() else str(device)
    cam = cv2.VideoCapture(source, cv2.CAP_V4L2)
    if cfg['fourcc']:
        cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cfg['fourcc']))
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, cfg['width'])
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg['height'])
    if not cam.isOpened():
        raise SystemExit(f'camera {device} could not be opened')
    for _ in range(cfg['warmup_frames']):  # let auto exposure settle before anything is saved
        cam.read()
    ok, frame = cam.read()
    if not ok or frame.shape[1::-1] != (cfg['width'], cfg['height']):
        raise SystemExit(f'camera must return {cfg["width"]}x{cfg["height"]}, got {None if not ok else frame.shape[1::-1]}')
    return cam, frame


def cam_props(cam):
    props = {p: cam.get(getattr(cv2, f'CAP_PROP_{p}')) for p in PROPS}
    fourcc = int(cam.get(cv2.CAP_PROP_FOURCC))
    props['FOURCC'] = ''.join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4))
    return props


def next_index(session_dir, stem):
    i = 1
    while (session_dir / f'{stem}_f{i:02d}_raw.png').exists():
        i += 1
    return i


def save_frame(session_dir, args, cfg, frame, calib, props, v4l2_text, device):
    session_dir.mkdir(parents=True, exist_ok=True)
    stem = f'{args.split}_{args.layout}'
    idx = next_index(session_dir, stem)
    base = session_dir / f'{stem}_f{idx:02d}'
    ok, png = cv2.imencode('.png', frame)
    assert ok
    raw_path = Path(f'{base}_raw.png')
    with open(raw_path, 'xb') as f:  # 'x' = refuse to overwrite
        f.write(png.tobytes())
    meta = {
        'timestamp': datetime.now(timezone.utc).astimezone().isoformat(),
        'session': args.session, 'split': args.split, 'layout': args.layout, 'frame_index': idx, 'note': args.note,
        'raw_file': raw_path.name, 'raw_sha256': sha256_bytes(png.tobytes()),
        'pixel_space': 'raw (distorted) camera pixels, BGR, x right / y down, origin top-left',
        'size': [frame.shape[1], frame.shape[0]],
        'device': str(device), 'device_resolved': str(Path(str(device)).resolve()) if not str(device).isdigit() else None,
        'cv2_props': props, 'v4l2_all': v4l2_text, 'capture_config': cfg,
        'undistorted_file': None,
    }
    if calib:
        und = cv2.remap(frame, *calib['maps'], cv2.INTER_LINEAR)
        ok, upng = cv2.imencode('.png', und)
        und_path = Path(f'{base}_undist.png')
        with open(und_path, 'xb') as f:
            f.write(upng.tobytes())
        meta.update({'undistorted_file': und_path.name, 'undistorted_sha256': sha256_bytes(upng.tobytes()),
                     'calibration_json': calib['path'], 'calibration_sha256': calib['sha256'],
                     'calibration_status': cfg.get('calibration_status'), 'undistort_convention': cfg.get('undistort_convention')})
    with open(f'{base}.json', 'x') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    with open(session_dir / 'manifest.jsonl', 'a') as f:
        f.write(json.dumps({k: meta[k] for k in ('timestamp', 'split', 'layout', 'frame_index', 'raw_file', 'raw_sha256')}) + '\n')
    return meta


def selftest():
    cfg = json.loads(CFG_PATH.read_text())
    cfg = dict(cfg, calibration_json=None)
    with tempfile.TemporaryDirectory() as tmp:
        args = argparse.Namespace(session='selftest', split='dev', layout='L01', note='selftest')
        frame = np.random.default_rng(0).integers(0, 255, (cfg['height'], cfg['width'], 3), dtype=np.uint8)
        d = Path(tmp) / 'selftest'
        m1 = save_frame(d, args, cfg, frame, None, {}, 'none', 'test')
        m2 = save_frame(d, args, cfg, frame, None, {}, 'none', 'test')
        assert (m1['frame_index'], m2['frame_index']) == (1, 2), 'indices must increment, never overwrite'
        back = cv2.imread(str(d / m1['raw_file']), cv2.IMREAD_UNCHANGED)
        assert np.array_equal(back, frame), 'PNG must be lossless'
        assert len((d / 'manifest.jsonl').read_text().splitlines()) == 2
    print('selftest PASS')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--session'); ap.add_argument('--split', choices=['dev', 'test'])
    ap.add_argument('--layout', help='layout id, e.g. L01'); ap.add_argument('--device'); ap.add_argument('--note', default='')
    ap.add_argument('--probe', action='store_true'); ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    cfg = json.loads(CFG_PATH.read_text())
    device = args.device or cfg['device']
    if not device:
        raise SystemExit('no device: pass --device /dev/v4l/by-id/... or set it in config/capture_config.json')
    apply_v4l2_controls(device, cfg['v4l2_controls_to_set'])
    cam, frame = open_camera(cfg, device)
    props, v4l2_text = cam_props(cam), v4l2_snapshot(device)
    if args.probe:
        print(json.dumps({'device': str(device), 'size': frame.shape[1::-1], 'props': props}, indent=2))
        print(v4l2_text)
        return cam.release()
    if not (args.session and args.split and args.layout):
        raise SystemExit('--session, --split and --layout are required for capture')
    calib = load_calibration(cfg['calibration_json'], cfg['width'], cfg['height'])
    session_dir = Path(cfg['out_root']) / args.session
    print(f'saving to {session_dir}  (SPACE = save, q/ESC = quit)')
    while True:
        ok, frame = cam.read()
        if not ok:  # camera failure: stop, do not save a stale frame
            raise SystemExit('camera read failed')
        view = cv2.resize(frame, (960, 540))
        cv2.putText(view, f'{args.split} {args.layout}  SPACE=save  q=quit', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow('CAPTURE (raw, preview 50%)', view)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        if key == 32:
            meta = save_frame(session_dir, args, cfg, frame, calib, cam_props(cam), v4l2_text, device)
            print('saved', meta['raw_file'], meta['raw_sha256'][:12])
    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    sys.exit(main())
