"""Deterministic fixed-reference pixel-space scene monitoring."""

from copy import deepcopy
import json
import math
from numbers import Real
from pathlib import Path

import cv2
import numpy as np


class SceneMonitorError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def _positive_int(value, name, allow_zero=False):
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SceneMonitorError("INVALID_CONFIG", f"{name} must be an integer >= {minimum}")
    return value


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise SceneMonitorError("INVALID_CONFIG", f"{name} must be finite")
    return float(value)


def validate_scene_monitor_config(config):
    if not isinstance(config, dict) or config.get("schema_version") != "0.1":
        raise SceneMonitorError("INVALID_CONFIG", "schema_version must be 0.1")
    selector = config.get("camera_selector")
    if not isinstance(selector, dict) or selector.get("policy") != "EXACT_EXTERNAL_BY_ID_ONLY":
        raise SceneMonitorError("INVALID_CONFIG", "exact external camera selection is required")
    for key in ("device_path", "required_identity_substring"):
        if not isinstance(selector.get(key), str) or not selector[key]:
            raise SceneMonitorError("INVALID_CONFIG", f"camera_selector.{key} is required")
    if selector.get("allow_fallback") is not False:
        raise SceneMonitorError("INVALID_CONFIG", "camera fallback must be disabled")
    forbidden = selector.get("forbidden_fallback_identity_substrings")
    if not isinstance(forbidden, list) or any(not isinstance(item, str) or not item for item in forbidden):
        raise SceneMonitorError("INVALID_CONFIG", "forbidden fallback identities must be strings")
    if any(item in selector["device_path"] for item in forbidden):
        raise SceneMonitorError("INVALID_CONFIG", "configured camera is a forbidden fallback")

    for key in (
        "preferred_width", "preferred_height", "warmup_frames", "maximum_reference_attempts",
        "reference_stability_required_count", "pixel_difference_threshold",
        "minimum_component_area_px",
    ):
        _positive_int(config.get(key), key, allow_zero=key == "warmup_frames")
    if config["pixel_difference_threshold"] > 255:
        raise SceneMonitorError("INVALID_CONFIG", "pixel_difference_threshold must be <= 255")
    if _finite(config.get("preferred_fps"), "preferred_fps") <= 0:
        raise SceneMonitorError("INVALID_CONFIG", "preferred_fps must be positive")
    for key in ("reference_stability_threshold_ratio", "minimum_component_area_ratio", "global_change_threshold"):
        value = _finite(config.get(key), key)
        if not 0 <= value <= 1:
            raise SceneMonitorError("INVALID_CONFIG", f"{key} must be within 0..1")
    if config.get("reference_mode") != "EXPLICIT_FIXED_REFERENCE":
        raise SceneMonitorError("INVALID_CONFIG", "reference_mode must be EXPLICIT_FIXED_REFERENCE")
    if config.get("parameter_status") != "DEVELOPMENT_PARAMETER":
        raise SceneMonitorError("INVALID_CONFIG", "parameters must be DEVELOPMENT_PARAMETER")

    preprocess = config.get("preprocess")
    kernel = preprocess.get("gaussian_kernel_size") if isinstance(preprocess, dict) else None
    _positive_int(kernel, "preprocess.gaussian_kernel_size")
    if kernel % 2 == 0:
        raise SceneMonitorError("INVALID_CONFIG", "Gaussian kernel size must be odd")
    morphology = config.get("morphology")
    if not isinstance(morphology, dict):
        raise SceneMonitorError("INVALID_CONFIG", "morphology config is required")
    morph_kernel = _positive_int(morphology.get("kernel_size"), "morphology.kernel_size")
    if morph_kernel % 2 == 0:
        raise SceneMonitorError("INVALID_CONFIG", "morphology kernel size must be odd")
    _positive_int(morphology.get("open_iterations"), "morphology.open_iterations", allow_zero=True)
    _positive_int(morphology.get("close_iterations"), "morphology.close_iterations", allow_zero=True)

    tracking = config.get("tracking")
    if not isinstance(tracking, dict):
        raise SceneMonitorError("INVALID_CONFIG", "tracking config is required")
    _positive_int(tracking.get("minimum_track_confirmations"), "tracking.minimum_track_confirmations")
    _positive_int(tracking.get("maximum_missing_frames"), "tracking.maximum_missing_frames", allow_zero=True)
    minimum_iou = _finite(tracking.get("minimum_iou"), "tracking.minimum_iou")
    distance = _finite(tracking.get("maximum_centroid_distance_px"), "tracking.maximum_centroid_distance_px")
    if not 0 <= minimum_iou <= 1 or distance < 0:
        raise SceneMonitorError("INVALID_CONFIG", "tracking thresholds are invalid")
    regions = config.get("ignore_regions_px")
    if not isinstance(regions, list):
        raise SceneMonitorError("INVALID_CONFIG", "ignore_regions_px must be a list")
    for region in regions:
        if not isinstance(region, list) or len(region) != 4 or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in region
        ) or region[0] >= region[2] or region[1] >= region[3]:
            raise SceneMonitorError("INVALID_CONFIG", "ignore regions must be ordered non-negative pixel boxes")
    live = config.get("live_validation")
    if not isinstance(live, dict):
        raise SceneMonitorError("INVALID_CONFIG", "live_validation config is required")
    _positive_int(live.get("maximum_frames"), "live_validation.maximum_frames")
    _positive_int(live.get("maximum_saved_sample_frames"), "live_validation.maximum_saved_sample_frames", allow_zero=True)
    if config.get("semantic_trigger_policy") != "CONFIRMED_LOCAL_PIXEL_TRACK_ONLY":
        raise SceneMonitorError("INVALID_CONFIG", "semantic trigger policy is invalid")
    return config


def load_scene_monitor_config(path):
    return validate_scene_monitor_config(json.loads(Path(path).read_text()))


def validate_frame(frame, expected_shape=None):
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise SceneMonitorError("INVALID_FRAME_SHAPE", "frame must be uint8 BGR with shape HxWx3")
    if frame.shape[0] <= 0 or frame.shape[1] <= 0:
        raise SceneMonitorError("INVALID_FRAME_SHAPE", "frame dimensions must be positive")
    if expected_shape is not None and frame.shape != expected_shape:
        raise SceneMonitorError("INVALID_FRAME_SHAPE", "frame shape differs from the fixed reference")
    return frame


def preprocess_frame(frame, config):
    validate_frame(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kernel = config["preprocess"]["gaussian_kernel_size"]
    return cv2.GaussianBlur(gray, (kernel, kernel), 0)


def _mask_ignored_regions(mask, regions):
    height, width = mask.shape
    for x_min, y_min, x_max, y_max in regions:
        if x_max > width or y_max > height:
            raise SceneMonitorError("INVALID_CONFIG", "ignore region exceeds frame bounds")
        mask[y_min:y_max, x_min:x_max] = 0


def detect_change_candidates(reference, frame, config, frame_index):
    validate_frame(reference)
    validate_frame(frame, reference.shape)
    reference_gray = preprocess_frame(reference, config)
    current_gray = preprocess_frame(frame, config)
    difference = cv2.absdiff(reference_gray, current_gray)
    _, mask = cv2.threshold(difference, config["pixel_difference_threshold"], 255, cv2.THRESH_BINARY)
    kernel_size = config["morphology"]["kernel_size"]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    if config["morphology"]["open_iterations"]:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=config["morphology"]["open_iterations"])
    if config["morphology"]["close_iterations"]:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=config["morphology"]["close_iterations"])
    _mask_ignored_regions(mask, config["ignore_regions_px"])

    pixel_count = int(np.count_nonzero(mask))
    total_pixels = mask.shape[0] * mask.shape[1]
    changed_ratio = pixel_count / total_pixels
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    components = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        components.append((area, label))
    largest = max((area for area, _ in components), default=0)
    score = {
        "changed_pixel_count": pixel_count,
        "changed_pixel_ratio": changed_ratio,
        "largest_component_area_px": largest,
        "largest_component_area_ratio": largest / total_pixels,
    }
    if changed_ratio >= config["global_change_threshold"]:
        return "GLOBAL_CHANGE", score, [], mask

    minimum_area = max(
        config["minimum_component_area_px"],
        math.ceil(config["minimum_component_area_ratio"] * total_pixels),
    )
    candidates = []
    for _, label in sorted(components, key=lambda item: (stats[item[1], cv2.CC_STAT_LEFT], stats[item[1], cv2.CC_STAT_TOP])):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < minimum_area:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        candidates.append({
            "candidate_index": len(candidates) + 1,
            "bbox_px": [x, y, x + width, y + height],
            "centroid_px": [float(centroids[label][0]), float(centroids[label][1])],
            "area_px": area,
            "area_ratio": area / total_pixels,
            "frame_index": frame_index,
            "source": "FAST_SCENE_CHANGE",
            "candidate_type": "CHANGE_CANDIDATE",
            "semantic_status": "NOT_EVALUATED",
            "metric_geometry_status": "NOT_AVAILABLE",
            "planning_role": "NOT_ASSIGNED",
        })
    return ("LOCAL_CHANGE" if candidates else "NO_SIGNIFICANT_CHANGE"), score, candidates, mask


def bbox_iou(first, second):
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    intersection = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return 0.0 if union == 0 else intersection / union


def centroid_distance(first, second):
    return math.hypot(first[0] - second[0], first[1] - second[1])


class ChangeTracker:
    def __init__(self, tracking_config):
        self.config = tracking_config
        self.tracks = {}
        self.retired_tracks = []
        self.next_track_number = 1

    def _new_track(self, candidate, frame_index):
        track_id = f"track_{self.next_track_number:03d}"
        self.next_track_number += 1
        track = {
            "track_id": track_id,
            "first_seen_frame": frame_index,
            "last_seen_frame": frame_index,
            "consecutive_seen_count": 1,
            "consecutive_missing_count": 0,
            "latest_bbox_px": deepcopy(candidate["bbox_px"]),
            "latest_centroid_px": deepcopy(candidate["centroid_px"]),
            "track_state": "TENTATIVE",
        }
        self.tracks[track_id] = track
        return track

    def update(self, candidates, frame_index):
        candidates = [deepcopy(item) for item in sorted(candidates, key=lambda item: item["candidate_index"])]
        pairs = []
        for track_id in sorted(self.tracks):
            track = self.tracks[track_id]
            for candidate_index, candidate in enumerate(candidates):
                iou = bbox_iou(track["latest_bbox_px"], candidate["bbox_px"])
                distance = centroid_distance(track["latest_centroid_px"], candidate["centroid_px"])
                if iou >= self.config["minimum_iou"] or distance <= self.config["maximum_centroid_distance_px"]:
                    pairs.append((-iou, distance, track_id, candidate_index))
        matched_tracks, matched_candidates = set(), set()
        assignments = {}
        for _, _, track_id, candidate_index in sorted(pairs):
            if track_id in matched_tracks or candidate_index in matched_candidates:
                continue
            matched_tracks.add(track_id)
            matched_candidates.add(candidate_index)
            assignments[candidate_index] = self.tracks[track_id]

        for candidate_index, candidate in enumerate(candidates):
            track = assignments.get(candidate_index)
            if track is None:
                track = self._new_track(candidate, frame_index)
            else:
                track["consecutive_seen_count"] = (
                    track["consecutive_seen_count"] + 1 if track["consecutive_missing_count"] == 0 else 1
                )
                track["consecutive_missing_count"] = 0
                track["last_seen_frame"] = frame_index
                track["latest_bbox_px"] = deepcopy(candidate["bbox_px"])
                track["latest_centroid_px"] = deepcopy(candidate["centroid_px"])
            track["track_state"] = (
                "CONFIRMED"
                if track["consecutive_seen_count"] >= self.config["minimum_track_confirmations"]
                else "TENTATIVE"
            )
            candidate["track_id"] = track["track_id"]
            candidate["track_state"] = track["track_state"]
            candidate["consecutive_seen_count"] = track["consecutive_seen_count"]

        for track_id in sorted(set(self.tracks) - matched_tracks - {item["track_id"] for item in assignments.values()}):
            track = self.tracks[track_id]
            if track["last_seen_frame"] == frame_index:
                continue
            track["consecutive_seen_count"] = 0
            track["consecutive_missing_count"] += 1
            track["track_state"] = "MISSING"
            if track["consecutive_missing_count"] > self.config["maximum_missing_frames"]:
                retired = deepcopy(track)
                retired["track_state"] = "RETIRED"
                self.retired_tracks.append(retired)
                del self.tracks[track_id]
        return candidates

    def snapshot(self):
        return [deepcopy(self.tracks[key]) for key in sorted(self.tracks)]


class SceneMonitor:
    def __init__(self, config, reference_frame, source):
        self.config = validate_scene_monitor_config(config)
        self.reference = validate_frame(reference_frame).copy()
        self.source = deepcopy(source)
        self.tracker = ChangeTracker(config["tracking"])
        self.latest_observation = None

    def process(self, frame, frame_index):
        if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index < 0:
            raise SceneMonitorError("INVALID_FRAME_INDEX", "frame_index must be non-negative")
        scene_state, score, candidates, _ = detect_change_candidates(
            self.reference, frame, self.config, frame_index
        )
        if scene_state == "GLOBAL_CHANGE":
            tracked = []
        else:
            tracked = self.tracker.update(candidates, frame_index)
        semantic_trigger = scene_state == "LOCAL_CHANGE" and any(
            item["track_state"] == "CONFIRMED" for item in tracked
        )
        observation = {
            "schema_version": "0.1",
            "frame_index": frame_index,
            "source": {
                "source_id": self.source["source_id"],
                "source_path": self.source["source_path"],
                "width": int(frame.shape[1]),
                "height": int(frame.shape[0]),
            },
            "scene_state": scene_state,
            "change_score": score,
            "candidates": tracked,
            "active_tracks": self.tracker.snapshot(),
            "tracking_scope": "PIXEL_SPACE_CHANGE_TRACKING",
            "semantic_trigger": semantic_trigger,
            "semantic_trigger_policy": self.config["semantic_trigger_policy"],
            "metric_conversion": "NOT_PERFORMED",
            "planner_injection": "NOT_PERFORMED",
        }
        self.latest_observation = deepcopy(observation)
        return observation

    def get_latest_trigger(self):
        if self.latest_observation is None or not self.latest_observation["semantic_trigger"]:
            return None
        return deepcopy(self.latest_observation)


def select_stable_reference(frames, config):
    previous = None
    stable_count = 0
    attempts = 0
    for frame in frames:
        attempts += 1
        if attempts > config["maximum_reference_attempts"]:
            break
        validate_frame(frame, None if previous is None else previous.shape)
        if previous is not None:
            first = preprocess_frame(previous, config)
            second = preprocess_frame(frame, config)
            changed = cv2.absdiff(first, second) > config["pixel_difference_threshold"]
            ratio = float(np.count_nonzero(changed)) / changed.size
            stable_count = stable_count + 1 if ratio <= config["reference_stability_threshold_ratio"] else 0
            if stable_count >= config["reference_stability_required_count"]:
                return frame.copy(), {"status": "REFERENCE_STABLE", "attempts": attempts, "last_changed_pixel_ratio": ratio}
        previous = frame
    raise SceneMonitorError("REFERENCE_NOT_STABLE", "reference stability was not reached within the attempt limit")
