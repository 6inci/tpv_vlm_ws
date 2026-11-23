"""Rectangular-footprint planning on the v0.2 NavMap, and an independent whole-body check.

v0.2 NavMap treats the robot as a disc. Here NavMap is reused unchanged and only subclassed: `valid` (robot-CENTRE configuration space) is replaced by
the erosion of the free mask with the footprint rectangle at a FIXED heading (a mecanum base translates without turning). A centre is valid only when
the whole rectangle lies on free cells; outside the map counts as blocked. The planner, goal selector and validators of v0.2 then run unchanged.
`sweep_check` does not use that grid logic: it re-rasterises the free space from the cm polygons at a finer resolution and tests the body rectangle
at every sample along the continuous waypoint polyline.
"""
import math

import cv2
import numpy as np

from semantic_task_nav.navigation.map_builder import NavMap


def footprint_dims_cm(length, width, margin, heading_deg):
    """(extent along x, extent along y) in cm of the margin-inflated rectangle; heading 0 = long axis along x, 90 = along y."""
    L, W = length + 2 * margin, width + 2 * margin
    return (L, W) if int(heading_deg) % 180 == 0 else (W, L)


class RectNavMap(NavMap):
    def __init__(self, spec, size, pcfg, extent_xy_cm, px_per_cm):
        ex, ey = extent_xy_cm
        self.extent_cm = (ex, ey)
        self.px_per_cm = px_per_cm
        super().__init__(spec, size, {**pcfg, "robot_radius_px": min(ex, ey) / 2 * px_per_cm})   # inscribed disc: used only for the soft risk cost
        kx, ky = 2 * math.ceil(ex / 2 * px_per_cm) + 1, 2 * math.ceil(ey / 2 * px_per_cm) + 1       # round OUTWARD: never smaller than the body
        fits = cv2.erode(self.free_full.astype(np.uint8), np.ones((ky, kx), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
        ys, xs = np.arange(self.rows) * self.cell_px + self.cell_px // 2, np.arange(self.cols) * self.cell_px + self.cell_px // 2
        self.footprint_fits_full = fits
        self.valid = fits[np.ix_(ys, xs)] & self.free
        self.risk = np.where(self.valid, np.clip(1.0 - (self.clearance - self.robot_radius_px) / self.safety_margin_px, 0.0, 1.0), np.nan)
        self.circumscribed_radius_px = math.hypot(ex, ey) / 2 * px_per_cm

    def summary(self):
        s = super().summary()
        s.update({"robot_footprint": f"rectangle {self.extent_cm[0]:.1f} x {self.extent_cm[1]:.1f} cm along x,y at a fixed heading (BEV px, {self.px_per_cm} px/cm)",
                  "frame_note": "BEV px of the PROVISIONAL floor transform; NOT raw_image_px (v0.2 labels in nested records say raw_image_px)"})
        s["cell_counts"]["valid_robot_center"] = int(self.valid.sum())
        return s

    def rotation_clear(self, xy_bev):
        """Conservative in-place rotation test: the circumscribed disc of the footprint must be on free pixels."""
        x, y = int(round(xy_bev[0])), int(round(xy_bev[1]))
        inside = 0 <= x < self.width and 0 <= y < self.height
        c = float(self.clearance_full[y, x]) if inside else 0.0
        return {"ok": bool(c >= self.circumscribed_radius_px), "clearance_px": round(c, 1), "needed_px": round(self.circumscribed_radius_px, 1)}


def free_mask_cm(spec_cm, size_cm, pad_cm, px_per_cm):
    """Independent rasterisation of free space = traversable - static - no_go, from the cm polygons, at px_per_cm."""
    w, h = round((size_cm[0] + 2 * pad_cm) * px_per_cm), round((size_cm[1] + 2 * pad_cm) * px_per_cm)

    def fill(polys):
        m = np.zeros((h, w), np.uint8)
        for p in polys:
            cv2.fillPoly(m, [np.round((np.asarray(p, np.float64) + pad_cm) * px_per_cm).astype(np.int32)], 1)
        return m.astype(bool)
    return fill(z["polygon"] for z in spec_cm["traversable"]) & ~fill(z["polygon"] for z in spec_cm["static_obstacles"]) & ~fill(z["polygon"] for z in spec_cm["no_go"])


def body_free(free, centre_cm, extent_cm, pad_cm, px_per_cm):
    """True when the rectangle centred at centre_cm lies entirely on free pixels (rounded outward; outside the canvas = blocked)."""
    x0, x1 = math.floor((centre_cm[0] - extent_cm[0] / 2 + pad_cm) * px_per_cm), math.ceil((centre_cm[0] + extent_cm[0] / 2 + pad_cm) * px_per_cm)
    y0, y1 = math.floor((centre_cm[1] - extent_cm[1] / 2 + pad_cm) * px_per_cm), math.ceil((centre_cm[1] + extent_cm[1] / 2 + pad_cm) * px_per_cm)
    if x0 < 0 or y0 < 0 or x1 > free.shape[1] or y1 > free.shape[0]:
        return False, int((max(0, -x0) + max(0, x1 - free.shape[1])) + (max(0, -y0) + max(0, y1 - free.shape[0])))
    bad = int((~free[y0:y1, x0:x1]).sum())
    return bad == 0, bad


def sweep_check(free, waypoints_cm, extent_cm, pad_cm, px_per_cm, step_cm):
    """Body rectangle (fixed heading) at every sample of the polyline (all vertices included)."""
    n_samples, first_bad = 0, None
    for i, (a, b) in enumerate(zip(waypoints_cm, waypoints_cm[1:])):
        n = max(1, int(math.ceil(math.dist(a, b) / step_cm)))
        for k in range(n + 1):
            t = k / n
            c = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
            ok, bad = body_free(free, c, extent_cm, pad_cm, px_per_cm)
            n_samples += 1
            if not ok and first_bad is None:
                first_bad = {"segment_index": i, "xy_cm": [round(c[0], 2), round(c[1], 2)], "non_free_px": bad}
    return {"ok": first_bad is None, "n_samples": n_samples, "step_cm": step_cm, "first_violation": first_bad,
            "check": "body rectangle on free pixels at every sample; free space re-rasterised from the cm polygons at %g px/cm" % px_per_cm}


def body_gap_cm(poly_cm, centre_cm, extent_cm, size_cm, pad_cm, px_per_cm):
    """Smallest distance between the NOMINAL body rectangle at centre_cm and a target polygon (cm), by distance transform."""
    w, h = round((size_cm[0] + 2 * pad_cm) * px_per_cm), round((size_cm[1] + 2 * pad_cm) * px_per_cm)
    m = np.ones((h, w), np.uint8)
    cv2.fillPoly(m, [np.round((np.asarray(poly_cm, np.float64) + pad_cm) * px_per_cm).astype(np.int32)], 0)
    dt = cv2.distanceTransform(m, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    x0, x1 = max(0, math.floor((centre_cm[0] - extent_cm[0] / 2 + pad_cm) * px_per_cm)), min(w, math.ceil((centre_cm[0] + extent_cm[0] / 2 + pad_cm) * px_per_cm))
    y0, y1 = max(0, math.floor((centre_cm[1] - extent_cm[1] / 2 + pad_cm) * px_per_cm)), min(h, math.ceil((centre_cm[1] + extent_cm[1] / 2 + pad_cm) * px_per_cm))
    return float(dt[y0:y1, x0:x1].min() / px_per_cm)
