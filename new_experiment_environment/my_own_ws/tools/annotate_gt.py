#!/usr/bin/env python3
"""Human answer-key (GT) box annotation. The GT is used only for scoring; it is never given to any model.

Usage:
  python annotate_gt.py --image IMG.png --registry object_registry_draft.json --annotator AB [--init-from PREV.gt.json] [--out OUT.gt.json]
  python annotate_gt.py --selftest
Mouse:  left-drag = new box with current label | left-click = select box | right-drag = pan | wheel = zoom
        click a label in the right panel = choose current label
Keys:   [ / ] prev/next label   r = relabel selected box   x = delete selected box   u = undo
        + / - zoom   f = fit view   t = toggle box text   s = save   q = quit (press twice if unsaved)
Boxes are stored as [x1, y1, x2, y2] float pixels of the image file as loaded (x right, y down, origin top-left).
"""
import argparse
import copy
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

PANEL_W, ROW_H = 300, 26
IGNORE = 'IGNORE'


def build_labels(registry):
    """Registry classes -> flat label list; classes with 'variants' become class:variant (e.g. drum:blue)."""
    labels = []
    for c in registry['classes']:
        labels += [f"{c['key']}:{v}" for v in c['variants']] if c.get('variants') else [c['key']]
    return labels + [IGNORE]


class View:
    """Display <-> image coordinate mapping. display = (image - offset) * scale."""

    def __init__(self, img_w, img_h, max_w, max_h):
        self.W, self.H = img_w, img_h
        self.fit = min(max_w / img_w, max_h / img_h)
        self.cw, self.ch = round(img_w * self.fit), round(img_h * self.fit)
        self.reset()

    def reset(self):
        self.s, self.ox, self.oy = self.fit, 0.0, 0.0

    def to_img(self, x, y):
        return self.ox + x / self.s, self.oy + y / self.s

    def to_disp(self, x, y):
        return (x - self.ox) * self.s, (y - self.oy) * self.s

    def clamp(self):
        self.ox = min(max(self.ox, 0.0), max(self.W - self.cw / self.s, 0.0))
        self.oy = min(max(self.oy, 0.0), max(self.H - self.ch / self.s, 0.0))

    def zoom(self, factor, cx, cy):
        px, py = self.to_img(cx, cy)
        self.s = min(max(self.s * factor, self.fit), 12.0)
        self.ox, self.oy = px - cx / self.s, py - cy / self.s
        self.clamp()

    def render(self, img):
        m = np.float32([[self.s, 0, -self.ox * self.s], [0, self.s, -self.oy * self.s]])
        interp = cv2.INTER_NEAREST if self.s >= 1 else cv2.INTER_LINEAR
        return cv2.warpAffine(img, m, (self.cw, self.ch), flags=interp)


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save_gt(path, doc):
    path = Path(path)
    doc['updated'] = datetime.now(timezone.utc).astimezone().isoformat()
    if path.exists():  # keep the previous version
        os.replace(path, path.with_suffix(path.suffix + '.bak'))
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    with os.fdopen(fd, 'w') as f:
        json.dump(doc, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def new_doc(image, registry_path, annotator, init_from, size):
    return {'schema': 'gt_boxes_v1', 'image': str(Path(image).resolve()), 'image_sha256': sha256_file(image),
            'image_size': list(size),
            'coordinate_system': 'pixels of the image file as loaded; [x1,y1,x2,y2]; x right, y down, origin top-left',
            'registry': str(Path(registry_path).resolve()), 'registry_sha256': sha256_file(registry_path),
            'annotator': annotator, 'created': datetime.now(timezone.utc).astimezone().isoformat(),
            'init_from': str(init_from) if init_from else None, 'objects': []}


def obj(label, box):
    x1, y1, x2, y2 = box
    cls, _, attr = label.partition(':')
    return {'label': label, 'class': cls, 'attribute': attr or None,
            'bbox_px': [round(min(x1, x2), 1), round(min(y1, y2), 1), round(max(x1, x2), 1), round(max(y1, y2), 1)]}


def color(label):
    if label == IGNORE:
        return (128, 128, 128)
    h = int(hashlib.md5(label.encode()).hexdigest()[:6], 16)
    return ((h >> 16) & 255 | 64, (h >> 8) & 255 | 64, h & 255 | 64)


def run_gui(args):
    registry = json.loads(Path(args.registry).read_text())
    labels = build_labels(registry)
    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f'cannot read {args.image}')
    out = Path(args.out or Path(args.image).with_suffix('.gt.json'))
    if out.exists():
        doc = json.loads(out.read_text())
        if doc['image_sha256'] != sha256_file(args.image):
            raise SystemExit(f'{out} belongs to a different image (sha256 mismatch)')
        print('continuing', out)
    else:
        doc = new_doc(args.image, args.registry, args.annotator, args.init_from, img.shape[1::-1])
        if args.init_from:  # start from a previous human-made GT (same fixed camera); moved objects must be corrected
            prev = json.loads(Path(args.init_from).read_text())
            if prev['image_size'] != doc['image_size']:
                raise SystemExit('init-from GT has a different image size')
            doc['objects'] = [obj(o['label'], o['bbox_px']) for o in prev['objects']]
            print(f'initialised {len(doc["objects"])} boxes from {args.init_from} - check every moved object')
    unknown = {o['label'] for o in doc['objects']} - set(labels)
    if unknown:
        raise SystemExit(f'labels not in registry: {unknown}')

    v = View(img.shape[1], img.shape[0], args.max_w - PANEL_W, args.max_h)
    st = {'cur': 0, 'sel': None, 'drag': None, 'pan': None, 'mouse': (0, 0), 'undo': [], 'dirty': False,
          'text': True, 'quit_armed': False}
    win = 'GT annotation'
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    def push_undo():
        st['undo'].append(copy.deepcopy(doc['objects']))
        st['dirty'], st['quit_armed'] = True, False

    def hit(ix, iy):
        cands = [(i, (o['bbox_px'][2] - o['bbox_px'][0]) * (o['bbox_px'][3] - o['bbox_px'][1]))
                 for i, o in enumerate(doc['objects'])
                 if o['bbox_px'][0] <= ix <= o['bbox_px'][2] and o['bbox_px'][1] <= iy <= o['bbox_px'][3]]
        return min(cands, key=lambda c: c[1])[0] if cands else None

    def on_mouse(event, x, y, flags, _):
        st['mouse'] = (min(x, v.cw - 1), y)
        if x >= v.cw:
            if event == cv2.EVENT_LBUTTONDOWN and 0 <= (y - 40) // ROW_H < len(labels):
                st['cur'] = (y - 40) // ROW_H
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            st['drag'] = (x, y, x, y)
        elif event == cv2.EVENT_MOUSEMOVE and st['drag']:
            st['drag'] = st['drag'][:2] + (x, y)
        elif event == cv2.EVENT_LBUTTONUP and st['drag']:
            x0, y0 = st['drag'][:2]
            st['drag'] = None
            if abs(x - x0) < 4 and abs(y - y0) < 4:
                st['sel'] = hit(*v.to_img(x, y))
            else:
                (ax, ay), (bx, by) = v.to_img(x0, y0), v.to_img(x, y)
                box = [min(max(c, 0), lim) for c, lim in zip((ax, ay, bx, by), (v.W, v.H, v.W, v.H))]
                push_undo()
                doc['objects'].append(obj(labels[st['cur']], box))
                st['sel'] = len(doc['objects']) - 1
        elif event == cv2.EVENT_RBUTTONDOWN:
            st['pan'] = (x, y, v.ox, v.oy)
        elif event == cv2.EVENT_MOUSEMOVE and st['pan']:
            x0, y0, ox0, oy0 = st['pan']
            v.ox, v.oy = ox0 - (x - x0) / v.s, oy0 - (y - y0) / v.s
            v.clamp()
        elif event == cv2.EVENT_RBUTTONUP:
            st['pan'] = None
        elif event == cv2.EVENT_MOUSEWHEEL:
            v.zoom(1.25 if cv2.getMouseWheelDelta(flags) > 0 else 0.8, x, y)

    cv2.setMouseCallback(win, on_mouse)
    while True:
        canvas = v.render(img)
        for i, o in enumerate(doc['objects']):
            (x1, y1), (x2, y2) = v.to_disp(*o['bbox_px'][:2]), v.to_disp(*o['bbox_px'][2:])
            c = color(o['label'])
            cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), c, 3 if i == st['sel'] else 1)
            if st['text']:
                cv2.putText(canvas, o['label'], (int(x1), int(y1) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)
        if st['drag']:
            cv2.rectangle(canvas, st['drag'][:2], st['drag'][2:], color(labels[st['cur']]), 1)
        panel = np.full((v.ch, PANEL_W, 3), 30, np.uint8)
        counts = {}
        for o in doc['objects']:
            counts[o['label']] = counts.get(o['label'], 0) + 1
        cv2.putText(panel, f'zoom x{v.s / v.fit:.1f}  boxes {len(doc["objects"])}{" *" if st["dirty"] else ""}',
                    (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        for i, lab in enumerate(labels):
            y = 40 + i * ROW_H
            if i == st['cur']:
                cv2.rectangle(panel, (0, y), (PANEL_W, y + ROW_H - 2), (90, 90, 90), -1)
            cv2.putText(panel, f'{lab} ({counts.get(lab, 0)})', (8, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color(lab), 1)
        cv2.imshow(win, np.hstack([canvas, panel]))
        k = cv2.waitKey(20) & 0xFF
        if k == 255:
            continue
        c = chr(k)
        if c == 'q':
            if st['dirty'] and not st['quit_armed']:
                st['quit_armed'] = True
                print('unsaved changes - press s to save, or q again to quit without saving')
                continue
            break
        st['quit_armed'] = False
        if c == ']':
            st['cur'] = (st['cur'] + 1) % len(labels)
        elif c == '[':
            st['cur'] = (st['cur'] - 1) % len(labels)
        elif c == 'x' and st['sel'] is not None:
            push_undo(); doc['objects'].pop(st['sel']); st['sel'] = None
        elif c == 'r' and st['sel'] is not None:
            push_undo(); doc['objects'][st['sel']] = obj(labels[st['cur']], doc['objects'][st['sel']]['bbox_px'])
        elif c == 'u' and st['undo']:
            doc['objects'] = st['undo'].pop(); st['sel'] = None; st['dirty'] = True
        elif c in '+=':
            v.zoom(1.25, *st['mouse'])
        elif c == '-':
            v.zoom(0.8, *st['mouse'])
        elif c == 'f':
            v.reset()
        elif c == 't':
            st['text'] = not st['text']
        elif c == 's':
            save_gt(out, doc); st['dirty'] = False
            print(f'saved {out} ({len(doc["objects"])} boxes)')
    cv2.destroyAllWindows()


def selftest():
    reg = {'classes': [{'key': 'drum', 'variants': ['blue', 'red']}, {'key': 'truck'}]}
    assert build_labels(reg) == ['drum:blue', 'drum:red', 'truck', IGNORE]
    v = View(1920, 1080, 1600, 900)
    v.zoom(3.0, 400, 300)
    for p in [(0, 0), (123.4, 567.8), (1919, 1079)]:
        assert np.allclose(v.to_img(*v.to_disp(*p)), p), 'display<->image round trip'
    assert np.allclose(v.to_img(400, 300), View(1920, 1080, 1600, 900).to_img(400, 300)), 'zoom keeps the cursor point fixed'
    o = obj('drum:blue', [50, 80, 10, 20])
    assert o['bbox_px'] == [10, 20, 50, 80] and o['class'] == 'drum' and o['attribute'] == 'blue'
    with tempfile.TemporaryDirectory() as tmp:
        img, reg_p = Path(tmp) / 'a.png', Path(tmp) / 'r.json'
        cv2.imwrite(str(img), np.zeros((10, 20, 3), np.uint8)); reg_p.write_text(json.dumps(reg))
        doc = new_doc(img, reg_p, 'T', None, (20, 10)); doc['objects'].append(o)
        out = Path(tmp) / 'a.gt.json'
        save_gt(out, doc); save_gt(out, doc)
        assert json.loads(out.read_text())['objects'] == [o] and out.with_suffix('.json.bak').exists()
    print('selftest PASS')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--image'); ap.add_argument('--registry'); ap.add_argument('--annotator')
    ap.add_argument('--init-from'); ap.add_argument('--out')
    ap.add_argument('--max-w', type=int, default=1900); ap.add_argument('--max-h', type=int, default=980)
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif not (a.image and a.registry and a.annotator):
        sys.exit('--image, --registry and --annotator are required')
    else:
        run_gui(a)
