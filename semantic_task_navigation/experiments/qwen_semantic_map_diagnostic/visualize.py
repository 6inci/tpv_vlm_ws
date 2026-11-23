"""Draws the model's own output on the original image. Nothing is corrected: boxes are the model's boxes (0-1000 -> pixels by
plain scaling), labels are the model's labels. Three images + review sheets:
  01_object_map.png         original + object boxes (solid, thin) with id + name, colour = the model's mobility class
  02_semantic_area_map.png  original + semantic-area boxes (dashed, translucent fill, own label) + links to the objects the
                            model relates to the area + the area the model proposed as destination
  03_comparison.png         original | object map | semantic-area map side by side, each with its source tag
  review/review_objects_N.png  crops of every drawn object box, for the reviewer's visual check (not model output)
Object boxes and area boxes are different data and use different line styles, so they cannot be mistaken for one another.
Text is ASCII only (OpenCV); non-ASCII characters in model strings are shown as '?'.
"""
import textwrap
from pathlib import Path

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
MOBILITY_COLOR = {"mobile_robot": (255, 0, 255), "other_vehicle": (0, 165, 255), "static_equipment_or_goods": (230, 200, 40),
                  "fixed_structure": (200, 110, 60), "uncertain": (170, 170, 170)}
AREA_COLORS = [(60, 220, 60), (255, 120, 60), (60, 200, 255), (200, 60, 255), (255, 255, 60), (60, 255, 200), (120, 120, 255),
               (255, 160, 200), (160, 255, 120), (255, 200, 120)]
SOURCE_TAG = "SOURCE: Qwen3-VL-4B-Instruct output, unmodified and unverified (no detector, no human correction)"


def _a(s):
    return str(s).encode("ascii", "replace").decode()


def _px(b, w, h):
    return [int(round(b[0] / 1000 * w)), int(round(b[1] / 1000 * h)), int(round(b[2] / 1000 * w)), int(round(b[3] / 1000 * h))]


def _label(img, text, org, fg, scale=0.5, bg=(0, 0, 0)):
    text = _a(text)
    (tw, th), bl = cv2.getTextSize(text, FONT, scale, 1)
    x = int(max(0, min(org[0], img.shape[1] - tw - 2)))
    y = int(max(th + 3, min(org[1], img.shape[0] - bl - 2)))
    cv2.rectangle(img, (x - 1, y - th - 2), (x + tw + 1, y + bl), bg, -1)
    cv2.putText(img, text, (x, y), FONT, scale, fg, 1, cv2.LINE_AA)


def _dashed_rect(img, b, color, thick, dash=16):
    x1, y1, x2, y2 = b
    for (p, q) in (((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)), ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))):
        p, q = np.array(p, float), np.array(q, float)
        n = max(1, int(np.linalg.norm(q - p) // dash))
        for i in range(0, n, 2):
            a, c = p + (q - p) * i / n, p + (q - p) * min(i + 1, n) / n
            cv2.line(img, tuple(a.astype(int)), tuple(c.astype(int)), (0, 0, 0), thick + 2, cv2.LINE_AA)
            cv2.line(img, tuple(a.astype(int)), tuple(c.astype(int)), color, thick, cv2.LINE_AA)


def _panel(width, lines, line_h=24):
    panel = np.full((line_h * len(lines) + 10, width, 3), 24, np.uint8)
    for i, (txt, col, swatch) in enumerate(lines):
        y = line_h * (i + 1) - 5
        x = 8
        if swatch is not None:
            cv2.rectangle(panel, (8, y - 12), (22, y + 1), swatch, -1)
            x = 30
        cv2.putText(panel, _a(txt)[:230], (x, y), FONT, 0.52, col, 1, cv2.LINE_AA)
    return panel


def _wrap(txt, col, swatch=None, width=205):
    return [((("   " if i else "") + p), col, swatch if i == 0 else None) for i, p in enumerate(textwrap.wrap(_a(txt), width) or [""])]


def object_map(image, usable_objs, meta):
    h, w = image.shape[:2]
    img = image.copy()
    for o in sorted(usable_objs, key=lambda o: -((o["bbox_2d"][2] - o["bbox_2d"][0]) * (o["bbox_2d"][3] - o["bbox_2d"][1]))):
        b, col = _px(o["bbox_2d"], w, h), MOBILITY_COLOR.get(o.get("mobility"), (170, 170, 170))
        t = 4 if o.get("mobility") == "mobile_robot" else 2
        cv2.rectangle(img, tuple(b[:2]), tuple(b[2:]), (0, 0, 0), t + 2)
        cv2.rectangle(img, tuple(b[:2]), tuple(b[2:]), col, t)
    for o in usable_objs:
        b, col = _px(o["bbox_2d"], w, h), MOBILITY_COLOR.get(o.get("mobility"), (170, 170, 170))
        _label(img, f"{o['id']} {o['name'][:22]}", (b[0] + 2, b[1] + 13), col, 0.42)
    counts = {m: sum(1 for o in usable_objs if o.get("mobility") == m) for m in MOBILITY_COLOR}
    lines = [(SOURCE_TAG, (255, 255, 255), None),
             (f"01 OBJECT MAP: {len(usable_objs)} object boxes drawn (solid lines). Boxes and mobility classes are the model's own; "
              "mobility is its visual impression, not a driveability fact.", (0, 255, 255), None)]
    lines += [(f"{m}: {n}", (230, 230, 230), c) for (m, c), n in zip(MOBILITY_COLOR.items(), counts.values())]
    lines += _wrap(f"run {meta['run_id']} | parse {meta['parse_status']} | image {meta['image_name']} (undistorted pixels, not aligned with any driving map)",
                   (170, 170, 170))
    return np.vstack([img, _panel(w, lines)])


def semantic_area_map(image, usable_objs, usable_areas, destination, meta):
    h, w = image.shape[:2]
    img = image.copy()
    by_id = {o["id"]: o for o in usable_objs}
    dest_id = (destination or {}).get("proposed_area_id") if (destination or {}).get("decision") == "proposed" else None
    layer = img.copy()
    for k, a in enumerate(usable_areas):
        b = _px(a["bbox_2d"], w, h)
        cv2.rectangle(layer, tuple(b[:2]), tuple(b[2:]), AREA_COLORS[k % len(AREA_COLORS)], -1)
    img = cv2.addWeighted(layer, 0.16, img, 0.84, 0)
    links = img.copy()
    for k, a in enumerate(usable_areas):
        col, b = AREA_COLORS[k % len(AREA_COLORS)], _px(a["bbox_2d"], w, h)
        c0 = ((b[0] + b[2]) // 2, (b[1] + b[3]) // 2)
        for oid in a.get("related_objects", []) or []:
            o = by_id.get(oid)
            if o:
                ob = _px(o["bbox_2d"], w, h)
                oc = ((ob[0] + ob[2]) // 2, (ob[1] + ob[3]) // 2)
                cv2.line(links, c0, oc, col, 1, cv2.LINE_AA)
                cv2.circle(links, oc, 5, col, -1, cv2.LINE_AA)
                cv2.circle(links, oc, 5, (0, 0, 0), 1, cv2.LINE_AA)
    img = cv2.addWeighted(links, 0.75, img, 0.25, 0)
    for k, a in enumerate(usable_areas):
        col, b = AREA_COLORS[k % len(AREA_COLORS)], _px(a["bbox_2d"], w, h)
        is_dest = a["id"] == dest_id
        _dashed_rect(img, b, (255, 255, 255) if is_dest else col, 7 if is_dest else 3)
        if is_dest:
            _dashed_rect(img, [b[0] + 5, b[1] + 5, b[2] - 5, b[3] - 5], (0, 0, 255), 3)
        _label(img, ("[DEST] " if is_dest else "") + f"[{a['id']}] {a['function_name'][:34]}", (b[0] + 4, b[1] + 22), col, 0.6)
    lines = [(SOURCE_TAG, (255, 255, 255), None),
             (f"02 SEMANTIC AREA MAP: {len(usable_areas)} area boxes (dashed, translucent). An area is a region the model assigned a function to; "
              "it is NOT an object and NOT an obstacle. Dots/lines = objects the model relates to the area.", (0, 255, 255), None)]
    for k, a in enumerate(usable_areas):
        rel = ",".join(a.get("related_objects", []) or [])
        lines += _wrap(f"{'[DEST] ' if a['id'] == dest_id else ''}{a['id']}: {a['function_name']} | basis {a.get('basis')} | "
                       f"uncertainty {a.get('uncertainty')} | related objects: {rel or '-'}", (230, 230, 230), AREA_COLORS[k % len(AREA_COLORS)])
    d = destination or {}
    lines += _wrap(f"destination decision: {d.get('decision')} | proposed area: {d.get('proposed_area_id')} | purpose as read by the model: "
                   f"{d.get('interpreted_purpose')}", (0, 200, 255))
    if d.get("decision") == "proposed" and dest_id not in {a['id'] for a in usable_areas}:
        lines += _wrap("the proposed area id is not among the drawable areas: nothing is highlighted", (120, 120, 255))
    return np.vstack([img, _panel(w, lines)])


def comparison(original, objmap, areamap, meta, col_w=900):
    cols = []
    for title, im, tag in (("ORIGINAL (undistorted photo, unmodified)", original, "source: camera frame"),
                           ("OBJECTS (model boxes, solid thin)", objmap, "source: Qwen3-VL output, unverified"),
                           ("SEMANTIC AREAS (model boxes, dashed)", areamap, "source: Qwen3-VL output, unverified")):
        h = int(im.shape[0] * col_w / im.shape[1])
        c = cv2.resize(im, (col_w, h), interpolation=cv2.INTER_AREA)
        head = np.full((54, col_w, 3), 24, np.uint8)
        cv2.putText(head, title, (8, 22), FONT, 0.62, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(head, tag, (8, 44), FONT, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        cols.append(np.vstack([head, c]))
    hmax = max(c.shape[0] for c in cols)
    cols = [cv2.copyMakeBorder(c, 0, hmax - c.shape[0], 0, 8, cv2.BORDER_CONSTANT, value=(0, 0, 0)) for c in cols]
    return np.hstack(cols)


def review_sheets(image, usable_objs, out_dir, per_sheet=30, cols=6, cell=(250, 190)):
    """Crops of each object box (with context) for the reviewer. Not model output."""
    h, w = image.shape[:2]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for s in range(0, len(usable_objs), per_sheet):
        chunk = usable_objs[s:s + per_sheet]
        rows = (len(chunk) + cols - 1) // cols
        sheet = np.full((rows * (cell[1] + 22), cols * cell[0], 3), 20, np.uint8)
        for k, o in enumerate(chunk):
            b = _px(o["bbox_2d"], w, h)
            pad = max(12, int(0.25 * max(b[2] - b[0], b[3] - b[1])))
            x1, y1, x2, y2 = max(0, b[0] - pad), max(0, b[1] - pad), min(w, b[2] + pad), min(h, b[3] + pad)
            c = image[y1:y2, x1:x2].copy()
            cv2.rectangle(c, (b[0] - x1, b[1] - y1), (b[2] - x1, b[3] - y1), (0, 255, 255), 1)
            sc = min(cell[0] / c.shape[1], cell[1] / c.shape[0])
            c = cv2.resize(c, (max(1, int(c.shape[1] * sc)), max(1, int(c.shape[0] * sc))), interpolation=cv2.INTER_CUBIC)
            r, cc = divmod(k, cols)
            y0, x0 = r * (cell[1] + 22), cc * cell[0]
            sheet[y0 + 22:y0 + 22 + c.shape[0], x0:x0 + c.shape[1]] = c
            cv2.putText(sheet, _a(f"{o['id']} {o['name']}")[:38], (x0 + 2, y0 + 15), FONT, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
        p = out_dir / f"review_objects_{s // per_sheet + 1}.png"
        cv2.imwrite(str(p), sheet)
        paths.append(p)
    return paths


def render_all(image, obj, out_dir, meta, usable_fn):
    out_dir = Path(out_dir)
    u, _ = usable_fn(obj)
    dest = (obj or {}).get("destination") if isinstance((obj or {}).get("destination"), dict) else None
    om = object_map(image, u["objects"], meta)
    am = semantic_area_map(image, u["objects"], u["semantic_areas"], dest, meta)
    paths = {"object_map": out_dir / "01_object_map.png", "semantic_area_map": out_dir / "02_semantic_area_map.png",
             "comparison": out_dir / "03_comparison.png"}
    cv2.imwrite(str(paths["object_map"]), om)
    cv2.imwrite(str(paths["semantic_area_map"]), am)
    cv2.imwrite(str(paths["comparison"]), comparison(image, om, am, meta))
    paths["review"] = review_sheets(image, u["objects"], out_dir / "review")
    return paths
