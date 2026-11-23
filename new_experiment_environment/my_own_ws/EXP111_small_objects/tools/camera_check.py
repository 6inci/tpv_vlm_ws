#!/usr/bin/env python3
"""EXP111 pre-capture check: is the camera where it was for EXP110?

Compares one new setup frame with the EXP110 reference image (dev P01) in raw distorted pixels:
  1. global image shift (cv2.phaseCorrelate on grayscale, px, + response)
  2. check image: EXP110 answer-key boxes of fixed objects drawn on the new frame, and a 50/50 blend
The user judges the result on site; the numbers are recorded with the capture notes.
Usage (qwen_vl env, from EXP111_small_objects):
  python tools/camera_check.py --new data/setup/S00_f00.png --out analysis/camera_check_<ts>
  python tools/camera_check.py --selftest
"""
import argparse, json, sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parents[1]
EXP110 = HERE.parent / 'EXP110_word_grounding'
REF_STEM = 'P01_cheese_2026-11-14-181534'
MOVABLE = ('small_pallet', 'block')


def shift(ref_bgr, new_bgr):
    a = np.float32(cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY))
    b = np.float32(cv2.cvtColor(new_bgr, cv2.COLOR_BGR2GRAY))
    win = cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(a, b, win)
    return dx, dy, resp


def selftest():
    ref = cv2.imread(str(EXP110 / 'data' / 'pilot' / f'{REF_STEM}.jpg'))
    m = np.float32([[1, 0, 7], [0, 1, -4]])
    moved = cv2.warpAffine(ref, m, (ref.shape[1], ref.shape[0]), borderMode=cv2.BORDER_REPLICATE)
    dx, dy, _ = shift(ref, moved)
    assert abs(dx - 7) < 0.5 and abs(dy + 4) < 0.5, (dx, dy)
    print(f'selftest PASS (known shift +7,-4 -> {dx:.2f},{dy:.2f})')


def main():
    if '--selftest' in sys.argv:
        return selftest()
    ap = argparse.ArgumentParser()
    ap.add_argument('--new', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    ref = cv2.imread(str(EXP110 / 'data' / 'pilot' / f'{REF_STEM}.jpg'))
    new = cv2.imread(a.new)
    if new is None or new.shape != ref.shape:
        raise SystemExit(f'new frame unreadable or size differs from the reference {ref.shape}')
    dx, dy, resp = shift(ref, new)
    gt = json.loads((EXP110 / 'annotations' / 'pilot' / f'{REF_STEM}.json').read_text())['objects']
    vis = new.copy()
    for o in gt:
        if o['label'] in MOVABLE:
            continue
        x1, y1, x2, y2 = map(int, o['box_xyxy'])
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.putText(vis, f'EXP110 fixed-object boxes on the new frame | shift dx={dx:.1f} dy={dy:.1f} px (response {resp:.2f})',
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    cv2.imwrite(str(out / 'fixed_boxes_on_new.jpg'), vis, [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(str(out / 'blend_ref_new.jpg'), cv2.addWeighted(ref, 0.5, new, 0.5, 0), [cv2.IMWRITE_JPEG_QUALITY, 92])
    res = {'reference': f'EXP110 data/pilot/{REF_STEM}.jpg', 'new': a.new, 'image_space': 'raw distorted px 1920x1080',
           'shift_px': {'dx': round(dx, 2), 'dy': round(dy, 2)}, 'phase_correlation_response': round(resp, 3),
           'note': 'global translation only; rotation or zoom changes show up as boxes that drift apart across the image'}
    (out / 'camera_check.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    sys.exit(main())
