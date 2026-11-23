#!/usr/bin/env python3
"""Answer-key (ground-truth) box tool for EXP110. Boxes are drawn by a person and are NEVER given to any model.

Usage (qwen_vl env):
  python tools/annotate.py data/dev/*.png --annotator binci
  python tools/annotate.py --selftest

Mouse : left-drag = new box with the current label | right-click = delete the smallest box under the cursor
Keys  : label hotkeys are listed in the 'legend' window
        s save | u undo last box | c copy all boxes from the previous image (then fix moved objects!)
        f toggle DONE | n next image | p previous image | q quit   (n/p/q also save)
Output: annotations/<image folder name>/<image stem>.json, boxes [x1, y1, x2, y2] in raw image pixels.
"""
import argparse
import hashlib
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / 'config.json').read_text())
ACFG = CFG['annotation']
KEYS = '1234567890abeghijklmortvwxyz'  # label hotkeys (s u c f n p q are commands)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_labels(registry):
    labels = [{'label': c['key'], 'variant': v} for c in registry['classes'] for v in (c.get('variants') or [None])]
    labels.append({'label': 'ignore', 'variant': None})
    if len(labels) > len(KEYS):
        raise SystemExit(f'{len(labels)} labels but only {len(KEYS)} hotkeys')
    return labels


def label_text(lab):
    return lab['label'] + (f":{lab['variant']}" if lab['variant'] else '')


def norm_box(x1, y1, x2, y2, w, h):
    """Display drag (any direction, image coords) -> [x1, y1, x2, y2] sorted and clipped to the image."""
    x1, x2 = sorted((min(max(x1, 0), w), min(max(x2, 0), w)))
    y1, y2 = sorted((min(max(y1, 0), h), min(max(y2, 0), h)))
    return [round(float(v), 1) for v in (x1, y1, x2, y2)]


def pick_box(objects, x, y):
    """Index of the smallest box containing (x, y), or None (smallest so nested boxes stay selectable)."""
    hits = [((o['box_xyxy'][2] - o['box_xyxy'][0]) * (o['box_xyxy'][3] - o['box_xyxy'][1]), i)
            for i, o in enumerate(objects)
            if o['box_xyxy'][0] <= x <= o['box_xyxy'][2] and o['box_xyxy'][1] <= y <= o['box_xyxy'][3]]
    return min(hits)[1] if hits else None


def ann_path(image, out_root):
    return Path(out_root) / Path(image).parent.name / f'{Path(image).stem}.json'


def new_record(image, w, h, registry_path, annotator):
    return {'experiment_id': CFG['experiment_id'], 'image': str(Path(image).resolve()), 'image_sha256': sha256(image),
            'image_size': [w, h], 'image_space': CFG['image_space'], 'registry': str(registry_path),
            'registry_sha256': sha256(registry_path), 'annotator': annotator, 'status': 'IN_PROGRESS',
            'copied_from': None, 'updated_at': None, 'tool': 'tools/annotate.py', 'objects': []}


def load_record(path, image):
    rec = json.loads(Path(path).read_text())
    if rec['image_sha256'] != sha256(image):
        raise SystemExit(f'{path} was made for a different image (sha256 mismatch) - refusing to load')
    return rec


def save_record(rec, path):
    rec['updated_at'] = datetime.now(timezone.utc).astimezone().isoformat()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(rec, indent=2, ensure_ascii=False))


def color(text):
    h = int(hashlib.md5(text.encode()).hexdigest()[:6], 16)
    return (h & 255 | 60, (h >> 8) & 255 | 60, (h >> 16) & 255 | 60)


class Annotator:
    def __init__(self, images, out_root, registry_path, annotator, scale):
        import cv2
        self.cv2 = cv2
        self.images, self.out_root, self.registry_path, self.annotator, self.s = images, out_root, registry_path, annotator, scale
        self.labels = build_labels(json.loads(Path(registry_path).read_text()))
        self.cur, self.idx, self.drag, self.cursor, self.msg = 0, 0, None, (0, 0), ''
        self.load(0)

    # ---- data ----
    def load(self, idx):
        self.idx = idx
        img = self.cv2.imread(str(self.images[idx]), self.cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit(f'cannot read {self.images[idx]}')
        self.img, (self.h, self.w) = img, img.shape[:2]
        self.path = ann_path(self.images[idx], self.out_root)
        self.rec = load_record(self.path, self.images[idx]) if self.path.exists() else \
            new_record(self.images[idx], self.w, self.h, self.registry_path, self.annotator)
        if self.rec['annotator'] != self.annotator:  # reviewing someone else's draft: keep both names
            self.rec.setdefault('draft_by', self.rec['annotator'])
            self.rec['reviewed_by'] = self.annotator
        self.redraw()

    def save(self):
        save_record(self.rec, self.path)
        self.msg = f'saved {self.path.name}'

    def copy_previous(self):
        if self.idx == 0:
            self.msg = 'no previous image'
            return
        prev = ann_path(self.images[self.idx - 1], self.out_root)
        if not prev.exists():
            self.msg = 'previous image has no saved annotation'
            return
        objs = json.loads(prev.read_text())['objects']
        self.rec['objects'] = [dict(o) for o in objs]
        self.rec['copied_from'] = str(prev)
        self.msg = f'copied {len(objs)} boxes - DELETE/REDRAW objects that moved'
        self.redraw()

    # ---- drawing ----
    def redraw(self):
        cv2, vis = self.cv2, self.img.copy()
        for o in self.rec['objects']:
            t = o['label'] + (f":{o['variant']}" if o.get('variant') else '')
            x1, y1, x2, y2 = map(int, o['box_xyxy'])
            cv2.rectangle(vis, (x1, y1), (x2, y2), (128, 128, 128) if o['ignore'] else color(t), 2)
            cv2.putText(vis, t, (x1 + 2, max(y1 - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color(t), 1, cv2.LINE_AA)
        self.full = vis

    def render(self):
        cv2, s = self.cv2, self.s
        disp = cv2.resize(self.full, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        if self.drag:
            (ax, ay), (bx, by) = self.drag
            cv2.rectangle(disp, (int(ax * s), int(ay * s)), (int(bx * s), int(by * s)), (0, 255, 255), 1)
        # magnifier around the cursor (full-resolution pixels)
        r, z = ACFG['magnifier_half_size_px'], ACFG['magnifier_zoom']
        cx, cy = self.cursor
        pad = cv2.copyMakeBorder(self.full, r, r, r, r, cv2.BORDER_CONSTANT)
        crop = pad[int(cy):int(cy) + 2 * r, int(cx):int(cx) + 2 * r]
        mag = cv2.resize(crop, (2 * r * z, 2 * r * z), interpolation=cv2.INTER_NEAREST)
        cv2.line(mag, (r * z, 0), (r * z, 2 * r * z), (0, 255, 255), 1)
        cv2.line(mag, (0, r * z), (2 * r * z, r * z), (0, 255, 255), 1)
        ox = 10 if cx * s > disp.shape[1] / 2 else disp.shape[1] - mag.shape[1] - 10
        disp[40:40 + mag.shape[0], ox:ox + mag.shape[1]] = mag
        lab = self.labels[self.cur]
        bar = (f"[{self.idx + 1}/{len(self.images)}] {Path(self.images[self.idx]).name} | label ({KEYS[self.cur]}): "
               f"{label_text(lab)} | boxes {len(self.rec['objects'])} | {self.rec['status']} | {self.msg}")
        cv2.rectangle(disp, (0, 0), (disp.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(disp, bar, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow('annotate', disp)

    def legend(self):
        cv2 = self.cv2
        rows = [f'{KEYS[i]}  {label_text(l)}' for i, l in enumerate(self.labels)]
        rows += ['', 'drag=box  right-click=delete', 's save  u undo  c copy prev', 'f DONE  n next  p prev  q quit']
        leg = np.full((24 * len(rows) + 16, 330, 3), 30, np.uint8)
        for i, t in enumerate(rows):
            col = (0, 255, 255) if i == self.cur else (230, 230, 230)
            cv2.putText(leg, t, (10, 24 * i + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)
        cv2.imshow('legend', leg)

    # ---- events ----
    def on_mouse(self, event, x, y, flags, param):
        cv2 = self.cv2
        ix, iy = x / self.s, y / self.s
        self.cursor = (min(max(ix, 0), self.w - 1), min(max(iy, 0), self.h - 1))
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag = [(ix, iy), (ix, iy)]
        elif event == cv2.EVENT_MOUSEMOVE and self.drag:
            self.drag[1] = (ix, iy)
        elif event == cv2.EVENT_LBUTTONUP and self.drag:
            (ax, ay), _ = self.drag
            self.drag = None
            box = norm_box(ax, ay, ix, iy, self.w, self.h)
            if box[2] - box[0] >= 2 and box[3] - box[1] >= 2:
                lab = self.labels[self.cur]
                self.rec['objects'].append({'label': lab['label'], 'variant': lab['variant'], 'box_xyxy': box,
                                            'ignore': lab['label'] == 'ignore'})
                self.msg = f'added {label_text(lab)}'
                self.redraw()
        elif event == cv2.EVENT_RBUTTONDOWN:
            i = pick_box(self.rec['objects'], ix, iy)
            if i is not None:
                o = self.rec['objects'].pop(i)
                self.msg = f"deleted {o['label']}"
                self.redraw()

    def run(self):
        cv2 = self.cv2
        cv2.namedWindow('annotate', cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback('annotate', self.on_mouse)
        while True:
            self.render()
            self.legend()
            k = cv2.waitKey(20)
            if k == -1:
                continue
            ch = chr(k & 0xFF)
            if ch in KEYS[:len(self.labels)]:
                self.cur = KEYS.index(ch)
            elif ch == 's':
                self.save()
            elif ch == 'u' and self.rec['objects']:
                self.rec['objects'].pop()
                self.msg = 'undo'
                self.redraw()
            elif ch == 'c':
                self.copy_previous()
            elif ch == 'f':
                self.rec['status'] = 'IN_PROGRESS' if self.rec['status'] == 'DONE' else 'DONE'
            elif ch in 'np':
                self.save()
                j = self.idx + (1 if ch == 'n' else -1)
                if 0 <= j < len(self.images):
                    self.load(j)
            elif ch == 'q' or k == 27:
                self.save()
                break
        cv2.destroyAllWindows()


def selftest():
    reg = json.loads(Path(CFG['registry']).read_text())
    labels = build_labels(reg)
    assert {'label': 'block', 'variant': 'blue'} in labels and labels[-1]['label'] == 'ignore', labels
    assert norm_box(50, 40, 10, 5, 30, 100) == [10.0, 5.0, 30.0, 40.0]  # reversed drag + clip at w=30
    objs = [{'box_xyxy': [0, 0, 100, 100]}, {'box_xyxy': [10, 10, 20, 20]}]
    assert pick_box(objs, 15, 15) == 1 and pick_box(objs, 50, 50) == 0 and pick_box(objs, 500, 5) is None
    with tempfile.TemporaryDirectory() as d:
        img = Path(d) / 'dev' / 'L01_f01.png'
        img.parent.mkdir()
        img.write_bytes(b'not really a png but hashable')
        rec = new_record(img, 1920, 1080, CFG['registry'], 'test')
        rec['objects'].append({'label': 'drum', 'variant': 'blue', 'box_xyxy': [1, 2, 3, 4], 'ignore': False})
        p = ann_path(img, Path(d) / 'ann')
        save_record(rec, p)
        assert p == Path(d) / 'ann' / 'dev' / 'L01_f01.json' and load_record(p, img)['objects'] == rec['objects']
        img.write_bytes(b'changed image')
        try:
            load_record(p, img)
            raise AssertionError('sha mismatch not detected')
        except SystemExit:
            pass
    print(f'selftest PASS ({len(labels)} labels)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('images', nargs='*')
    ap.add_argument('--annotator', default='unknown')
    ap.add_argument('--out', default=str(ROOT / CFG['paths']['annotations']))
    ap.add_argument('--scale', type=float, default=ACFG['display_scale'])
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.images:
        ap.error('give image paths')
    Annotator(sorted(a.images), a.out, CFG['registry'], a.annotator, a.scale).run()


if __name__ == '__main__':
    sys.exit(main())
