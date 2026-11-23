"""Explicit transforms between coordinate frames (see docs/COORDINATE_FRAMES.md).

raw_image_px  <->  roi_local_px : raw = offset + roi / scale      (crop + uniform zoom)
raw_image_px  <->  grid_row_col : cell centre = ((c + 0.5) * cell_px, (r + 0.5) * cell_px)
"""
import math

from ..contracts import GRID_RC, RAW_PX, ROI_PX


class RoiTransform:
    """Crop [x0, y0, x1, y1) of the raw image, resized uniformly by `scale`."""

    def __init__(self, crop_xyxy, scale):
        x0, y0, x1, y1 = (int(v) for v in crop_xyxy)
        if x1 <= x0 or y1 <= y0 or scale <= 0:
            raise ValueError(f"invalid ROI {crop_xyxy} scale {scale}")
        self.crop_xyxy, self.scale = [x0, y0, x1, y1], float(scale)

    @property
    def zoom_size(self):
        x0, y0, x1, y1 = self.crop_xyxy
        return round((x1 - x0) * self.scale), round((y1 - y0) * self.scale)

    def roi_to_raw_point(self, x, y):
        return self.crop_xyxy[0] + x / self.scale, self.crop_xyxy[1] + y / self.scale

    def raw_to_roi_point(self, x, y):
        return (x - self.crop_xyxy[0]) * self.scale, (y - self.crop_xyxy[1]) * self.scale

    def roi_to_raw_box(self, box):
        x1, y1 = self.roi_to_raw_point(box[0], box[1])
        x2, y2 = self.roi_to_raw_point(box[2], box[3])
        return [x1, y1, x2, y2]

    def raw_to_roi_box(self, box):
        x1, y1 = self.raw_to_roi_point(box[0], box[1])
        x2, y2 = self.raw_to_roi_point(box[2], box[3])
        return [x1, y1, x2, y2]

    def to_dict(self):
        return {"from_frame": ROI_PX, "to_frame": RAW_PX, "crop_xyxy_raw": self.crop_xyxy, "scale": self.scale,
                "zoom_size_wh": list(self.zoom_size), "formula": "raw = crop_offset + roi / scale"}


def expand_box(box, ratio, min_margin, width, height):
    """Expand by max(ratio * size, min_margin) on every side, clipped to the image (integers)."""
    x1, y1, x2, y2 = box
    mx = max(ratio * (x2 - x1), min_margin)
    my = max(ratio * (y2 - y1), min_margin)
    return [max(0, math.floor(x1 - mx)), max(0, math.floor(y1 - my)),
            min(width, math.ceil(x2 + mx)), min(height, math.ceil(y2 + my))]


def px_to_cell(x, y, cell_px):
    return int(math.floor(y / cell_px)), int(math.floor(x / cell_px))


def cell_to_px(row, col, cell_px):
    return (col + 0.5) * cell_px, (row + 0.5) * cell_px


def tagged_box(xyxy, frame=RAW_PX):
    return {"xyxy": [round(float(v), 2) for v in xyxy], "frame": frame}


def tagged_cell(row, col):
    return {"rc": [int(row), int(col)], "frame": GRID_RC}


def box_center(b):
    return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0


def box_area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def box_gap(a, b):
    """Euclidean gap between two axis-aligned boxes (0 if they overlap)."""
    dx = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    return math.hypot(dx, dy)


def containment(inner, outer):
    """Fraction of `inner` area lying inside `outer`."""
    ix = max(0.0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    iy = max(0.0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    a = box_area(inner)
    return ix * iy / a if a > 0 else 0.0
