"""Stored-image geometry audit and deterministic STEP 20 offline replay."""

import hashlib
import json
from pathlib import Path
import re

import cv2
import numpy as np

from .scene_monitor import SceneMonitor


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
COMPATIBLE_STATUSES = {"REFERENCE_VERIFIED", "DEVELOPMENT_COMPATIBLE_CANDIDATE"}
DEVELOPMENT_PROVENANCE = (
    "Image-registration-based development inference only; not final physical camera-pose validation."
)
DIAGNOSTIC_ROLE = "IMAGE_TO_IMAGE_COMPATIBILITY_DIAGNOSTIC_ONLY_NOT_METRIC_HOMOGRAPHY"


class ImageGeometryAuditError(ValueError):
    pass


def _json(path):
    return json.loads(Path(path).read_text())


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_audit_config(path):
    config = _json(path)
    required = (
        "reference_image", "candidate_directory", "allowed_categories", "registration_method",
        "opencv_rng_seed", "orb", "ratio_test", "ransac_reprojection_threshold_px",
        "minimum_accepted_matches", "minimum_inliers", "minimum_inlier_ratio",
        "maximum_reprojection_error_px", "maximum_corner_displacement_px", "parameter_status",
    )
    if config.get("schema_version") != "0.1" or any(key not in config for key in required):
        raise ImageGeometryAuditError("invalid offline image geometry audit config")
    if config["parameter_status"] != "DEVELOPMENT_PARAMETER":
        raise ImageGeometryAuditError("registration thresholds must be DEVELOPMENT_PARAMETER")
    if config["allowed_categories"] != ["RAW_LIKE_SOURCE"]:
        raise ImageGeometryAuditError("only RAW_LIKE_SOURCE images may enter replay")
    return config


def classify_image(path, reference_image):
    path, reference_image = Path(path).resolve(), Path(reference_image).resolve()
    name = path.name.lower()
    if path == reference_image or re.fullmatch(r"photo_\d+\.(png|jpe?g)", name):
        return "RAW_LIKE_SOURCE"
    if name == "qwen_frame.jpg":
        return "RAW_LIKE_SOURCE"
    if "undistorted" in name:
        return "UNDISTORTED_DERIVED"
    if "bbox" in name or "annotated" in name:
        return "ANNOTATED_DERIVED"
    if any(token in name for token in ("grid", "homography", "marked", "visualized", "visualization")):
        return "VISUALIZATION_DERIVED"
    return "UNKNOWN"


def _relation(path, reference_image, category):
    path, reference_image = Path(path).resolve(), Path(reference_image).resolve()
    if path == reference_image:
        return "LEGACY_GEOMETRY_REFERENCE"
    if category == "RAW_LIKE_SOURCE" and re.fullmatch(r"photo_\d+\.(png|jpe?g)", path.name.lower()):
        return "SAME_NAMED_CAPTURE_SERIES_CANDIDATE"
    if category == "RAW_LIKE_SOURCE":
        return "OTHER_RAW_LIKE_SOURCE_UNKNOWN_CAMERA_GEOMETRY"
    if path.name.lower().startswith("photo_04_"):
        return "DERIVED_FROM_PHOTO_04"
    return "NO_ESTABLISHED_RELATION"


def inventory_images(directory, reference_image):
    directory, reference_image = Path(directory).resolve(), Path(reference_image).resolve()
    paths = sorted(
        path for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    images = []
    for number, path in enumerate(paths, start=1):
        decoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if decoded is None:
            raise ImageGeometryAuditError(f"could not decode image: {path}")
        height, width = decoded.shape[:2]
        channels = 1 if decoded.ndim == 2 else decoded.shape[2]
        category = classify_image(path, reference_image)
        images.append({
            "image_id": f"image_{number:03d}_{path.stem}",
            "filename": path.name,
            "absolute_path": str(path.resolve()),
            "sha256": _sha256(path),
            "file_size": path.stat().st_size,
            "width": width,
            "height": height,
            "channels": channels,
            "dtype": str(decoded.dtype),
            "category": category,
            "relation_to_photo_04": _relation(path, reference_image, category),
        })
    return {
        "schema_version": "0.1",
        "source_directory": str(directory),
        "reference_image": str(reference_image),
        "image_count": len(images),
        "images": images,
    }


def read_bgr(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ImageGeometryAuditError(f"could not decode image: {path}")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.shape[2] != 3:
        raise ImageGeometryAuditError(f"unsupported channel count: {path}")
    return image


def orb_features(gray, config):
    params = config["orb"]
    score_type = cv2.ORB_HARRIS_SCORE if params["scoreType"] == "HARRIS" else cv2.ORB_FAST_SCORE
    cv2.setRNGSeed(config["opencv_rng_seed"])
    detector = cv2.ORB_create(
        nfeatures=params["nfeatures"], scaleFactor=params["scaleFactor"], nlevels=params["nlevels"],
        edgeThreshold=params["edgeThreshold"], firstLevel=params["firstLevel"], WTA_K=params["WTA_K"],
        scoreType=score_type, patchSize=params["patchSize"], fastThreshold=params["fastThreshold"],
    )
    return detector.detectAndCompute(gray, None)


def ordered_ratio_matches(candidate_descriptors, reference_descriptors, ratio):
    if candidate_descriptors is None or reference_descriptors is None:
        return 0, []
    raw = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(candidate_descriptors, reference_descriptors, k=2)
    accepted = [pair[0] for pair in raw if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance]
    return len(raw), sorted(accepted, key=lambda item: (item.distance, item.queryIdx, item.trainIdx))


def _empty_metrics(reference_count, candidate_count, status):
    return {
        "reference_keypoint_count": reference_count,
        "candidate_keypoint_count": candidate_count,
        "raw_match_count": 0,
        "accepted_match_count": 0,
        "inlier_count": 0,
        "inlier_ratio": None,
        "reprojection_error_px": None,
        "median_reprojection_error_px": None,
        "reference_corner_displacement_px": None,
        "mean_reference_corner_displacement_px": None,
        "registration_status": status,
    }


def classify_registration(metrics, resolution_status, config):
    if resolution_status == "RESOLUTION_MISMATCH":
        return "RESOLUTION_MISMATCH"
    if metrics["registration_status"] != "REGISTERED":
        return "INSUFFICIENT_EVIDENCE"
    sufficient = (
        metrics["accepted_match_count"] >= config["minimum_accepted_matches"]
        and metrics["inlier_count"] >= config["minimum_inliers"]
        and metrics["inlier_ratio"] >= config["minimum_inlier_ratio"]
    )
    if not sufficient:
        return "INSUFFICIENT_EVIDENCE"
    compatible = (
        metrics["reprojection_error_px"] <= config["maximum_reprojection_error_px"]
        and metrics["reference_corner_displacement_px"] <= config["maximum_corner_displacement_px"]
    )
    return "DEVELOPMENT_COMPATIBLE_CANDIDATE" if compatible else "POSE_MISMATCH"


def metric_transform_policy(status):
    return {
        "REFERENCE_VERIFIED": "REFERENCE_ALLOWED",
        "DEVELOPMENT_COMPATIBLE_CANDIDATE": "DEVELOPMENT_INFERRED_ALLOWED",
        "POSE_MISMATCH": "NOT_ALLOWED",
        "RESOLUTION_MISMATCH": "NOT_ALLOWED",
        "INSUFFICIENT_EVIDENCE": "UNKNOWN",
    }[status]


def _register(reference, reference_features, candidate, config):
    reference_keypoints, reference_descriptors = reference_features
    candidate_gray = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY)
    candidate_keypoints, candidate_descriptors = orb_features(candidate_gray, config)
    if candidate.shape[:2] != reference.shape[:2]:
        return _empty_metrics(len(reference_keypoints), len(candidate_keypoints), "RESOLUTION_MISMATCH"), None
    raw_count, matches = ordered_ratio_matches(
        candidate_descriptors, reference_descriptors, config["ratio_test"]
    )
    metrics = _empty_metrics(len(reference_keypoints), len(candidate_keypoints), "INSUFFICIENT_FEATURES")
    metrics["raw_match_count"] = raw_count
    metrics["accepted_match_count"] = len(matches)
    if len(matches) < 4:
        return metrics, None
    source = np.float32([candidate_keypoints[item.queryIdx].pt for item in matches])
    target = np.float32([reference_keypoints[item.trainIdx].pt for item in matches])
    cv2.setRNGSeed(config["opencv_rng_seed"])
    transform, mask = cv2.findHomography(
        source, target, cv2.RANSAC, config["ransac_reprojection_threshold_px"]
    )
    if transform is None or mask is None:
        metrics["registration_status"] = "HOMOGRAPHY_ESTIMATION_FAILED"
        return metrics, None
    inliers = mask.ravel().astype(bool)
    if not np.any(inliers):
        metrics["registration_status"] = "HOMOGRAPHY_ESTIMATION_FAILED"
        return metrics, None
    projected = cv2.perspectiveTransform(source.reshape(-1, 1, 2), transform).reshape(-1, 2)
    errors = np.linalg.norm(projected[inliers] - target[inliers], axis=1)
    height, width = reference.shape[:2]
    corners = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    projected_corners = cv2.perspectiveTransform(corners.reshape(1, -1, 2), transform)[0]
    displacements = np.linalg.norm(projected_corners - corners, axis=1)
    metrics.update({
        "inlier_count": int(np.count_nonzero(inliers)),
        "inlier_ratio": round(float(np.mean(inliers)), 9),
        "reprojection_error_px": round(float(np.mean(errors)), 9),
        "median_reprojection_error_px": round(float(np.median(errors)), 9),
        "reference_corner_displacement_px": round(float(np.max(displacements)), 9),
        "mean_reference_corner_displacement_px": round(float(np.mean(displacements)), 9),
        "registration_status": "REGISTERED",
    })
    diagnostic = {
        "candidate_keypoints": candidate_keypoints,
        "reference_keypoints": reference_keypoints,
        "matches": matches,
        "transform": transform,
    }
    return metrics, diagnostic


def audit_legacy_geometry(config):
    calibration_path = Path(config["calibration_path"]).resolve()
    geometry_path = Path(config["legacy_geometry_path"]).resolve()
    undistorted_path = Path(config["legacy_undistorted_reference"]).resolve()
    calibration, geometry = _json(calibration_path), _json(geometry_path)
    world = np.asarray(geometry["dst_world_xy"], dtype=np.float64)
    return {
        "status": "LEGACY_GEOMETRY_REFERENCE_READ_ONLY",
        "calibration": {"path": str(calibration_path), "sha256": _sha256(calibration_path)},
        "geometry": {"path": str(geometry_path), "sha256": _sha256(geometry_path)},
        "undistorted_reference": {"path": str(undistorted_path), "sha256": _sha256(undistorted_path)},
        "calibration_size": [calibration["image_width"], calibration["image_height"]],
        "legacy_image_size": [geometry["image_width"], geometry["image_height"]],
        "workspace_bounds_m": {
            "x": [float(np.min(world[:, 0])), float(np.max(world[:, 0]))],
            "y": [float(np.min(world[:, 1])), float(np.max(world[:, 1]))],
        },
        "required_pipeline": [
            "RAW_IMAGE_PX", "STORED_UNDISTORTION", "UNDISTORTED_IMAGE_PX",
            "LEGACY_HOMOGRAPHY", "LEGACY_WORKSPACE_COORDINATE",
        ],
        "raw_pixel_direct_to_legacy_homography": "FORBIDDEN",
        "metric_transform_executed": False,
    }


def audit_geometry(config, inventory, include_diagnostics=False):
    reference_path = str(Path(config["reference_image"]).resolve())
    by_path = {item["absolute_path"]: item for item in inventory["images"]}
    if reference_path not in by_path:
        raise ImageGeometryAuditError("reference image is missing from inventory")
    reference_record = by_path[reference_path]
    reference = read_bgr(reference_path)
    reference_features = orb_features(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), config)
    entries, diagnostics = [], {}
    raw_records = [item for item in inventory["images"] if item["category"] in config["allowed_categories"]]
    for record in sorted(raw_records, key=lambda item: item["filename"]):
        candidate = read_bgr(record["absolute_path"])
        resolution_status = (
            "SAME_RESOLUTION" if candidate.shape[:2] == reference.shape[:2] else "RESOLUTION_MISMATCH"
        )
        if record["absolute_path"] == reference_path:
            count = len(reference_features[0])
            metrics = {
                "reference_keypoint_count": count, "candidate_keypoint_count": count,
                "raw_match_count": count, "accepted_match_count": count, "inlier_count": count,
                "inlier_ratio": 1.0, "reprojection_error_px": 0.0,
                "median_reprojection_error_px": 0.0, "reference_corner_displacement_px": 0.0,
                "mean_reference_corner_displacement_px": 0.0, "registration_status": "REFERENCE_IDENTITY",
            }
            status, diagnostic = "REFERENCE_VERIFIED", None
        else:
            metrics, diagnostic = _register(reference, reference_features, candidate, config)
            status = classify_registration(metrics, resolution_status, config)
        policy = metric_transform_policy(status)
        entry = {
            "image_id": record["image_id"],
            "filename": record["filename"],
            "absolute_path": record["absolute_path"],
            "category": record["category"],
            "resolution_status": resolution_status,
            **metrics,
            "geometry_compatibility_status": status,
            "legacy_metric_transform_policy": policy,
            "metric_transform_provenance": DEVELOPMENT_PROVENANCE if policy == "DEVELOPMENT_INFERRED_ALLOWED" else None,
            "diagnostic_transform_role": DIAGNOSTIC_ROLE,
        }
        entries.append(entry)
        if include_diagnostics and diagnostic is not None:
            diagnostics[record["image_id"]] = diagnostic
    compatible = [item for item in entries if item["geometry_compatibility_status"] in COMPATIBLE_STATUSES]
    candidates = [item for item in compatible if item["geometry_compatibility_status"] != "REFERENCE_VERIFIED"]
    dataset_status = "OFFLINE_REPLAY_READY" if len(candidates) >= 2 else (
        "OFFLINE_REPLAY_PARTIAL" if compatible else "OFFLINE_REPLAY_NOT_READY"
    )
    return {
        "schema_version": "0.1",
        "reference_image_id": reference_record["image_id"],
        "reference_status": "LEGACY_GEOMETRY_REFERENCE",
        "registration_method": config["registration_method"],
        "parameter_status": config["parameter_status"],
        "same_resolution_is_compatibility_evidence": False,
        "static_region_bias_limit": "DEVELOPMENT_AUDIT_NOT_FINAL_PHYSICAL_CAMERA_POSE_VALIDATION",
        "images": entries,
        "compatible_candidate_count": len(candidates),
        "dataset_usability_status": dataset_status,
    }, diagnostics


def build_replay_manifest(config, audit):
    members = [
        item for item in audit["images"]
        if item["geometry_compatibility_status"] in COMPATIBLE_STATUSES
    ]
    images = [{
        "replay_index": index,
        "image_id": item["image_id"],
        "path": item["absolute_path"],
        "category": item["category"],
        "compatibility_status": item["geometry_compatibility_status"],
        "metric_transform_policy": item["legacy_metric_transform_policy"],
    } for index, item in enumerate(sorted(members, key=lambda item: item["filename"]), start=1)]
    return {
        "schema_version": "0.1",
        "group_id": "legacy_geometry_offline_group_01",
        "reference_image": str(Path(config["reference_image"]).resolve()),
        "image_domain": "RAW_LIKE_SOURCE_DECODED_TO_BGR_WITHOUT_RESIZE",
        "ordering_policy": "DETERMINISTIC_FILENAME_ORDER_NOT_REAL_CAPTURE_TIMELINE",
        "images": images,
        "geometry_status": audit["dataset_usability_status"],
        "scene_monitor_source": "EXISTING_STEP_20_SCENE_MONITOR",
        "purpose": "REAL_IMAGE_OFFLINE_DEVELOPMENT_REPLAY",
    }


def write_registration_visualizations(output_dir, reference_path, audit, diagnostics):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    reference = read_bgr(reference_path)
    by_id = {item["image_id"]: item for item in audit["images"]}
    for image_id, diagnostic in sorted(diagnostics.items()):
        entry = by_id[image_id]
        candidate = read_bgr(entry["absolute_path"])
        stem = Path(entry["filename"]).stem
        shown_matches = diagnostic["matches"][:80]
        feature_view = cv2.drawMatches(
            candidate, diagnostic["candidate_keypoints"], reference, diagnostic["reference_keypoints"],
            shown_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        height, width = reference.shape[:2]
        registered = cv2.warpPerspective(candidate, diagnostic["transform"], (width, height))
        overlay = cv2.addWeighted(reference, 0.5, registered, 0.5, 0)
        difference = cv2.absdiff(reference, registered)
        outputs = {
            f"{stem}__feature_matches.png": feature_view,
            f"{stem}__candidate_registered_to_reference.png": registered,
            f"{stem}__reference_candidate_overlay.png": overlay,
            f"{stem}__difference_after_registration.png": difference,
        }
        for filename, image in outputs.items():
            if not cv2.imwrite(str(output_dir / filename), image):
                raise ImageGeometryAuditError(f"failed to write visualization: {filename}")


def run_offline_replay(manifest, scene_monitor_config):
    if manifest.get("purpose") != "REAL_IMAGE_OFFLINE_DEVELOPMENT_REPLAY":
        raise ImageGeometryAuditError("invalid replay purpose")
    if manifest.get("ordering_policy") != "DETERMINISTIC_FILENAME_ORDER_NOT_REAL_CAPTURE_TIMELINE":
        raise ImageGeometryAuditError("replay order must not claim a real capture timeline")
    reference = read_bgr(manifest["reference_image"])
    monitor = SceneMonitor(scene_monitor_config, reference, {
        "source_id": manifest["group_id"], "source_path": "OFFLINE_STORED_IMAGE_MANIFEST",
    })
    observations = []
    for item in manifest["images"]:
        if item["category"] != "RAW_LIKE_SOURCE" or item["compatibility_status"] not in COMPATIBLE_STATUSES:
            raise ImageGeometryAuditError("manifest contains an ineligible replay image")
        frame = read_bgr(item["path"])
        if frame.shape != reference.shape:
            raise ImageGeometryAuditError("replay image differs from fixed reference; resizing is forbidden")
        raw = monitor.process(frame, item["replay_index"])
        candidates = [{
            "bbox_px": candidate["bbox_px"],
            "centroid_px": candidate["centroid_px"],
            "area_px": candidate["area_px"],
            "track_id": candidate.get("track_id"),
            "track_state": candidate.get("track_state"),
        } for candidate in raw["candidates"]]
        observations.append({
            "replay_index": item["replay_index"],
            "source_image": item["path"],
            "reference_image": manifest["reference_image"],
            "geometry_compatibility_status": item["compatibility_status"],
            "scene_state": raw["scene_state"],
            "changed_pixel_count": raw["change_score"]["changed_pixel_count"],
            "changed_pixel_ratio": raw["change_score"]["changed_pixel_ratio"],
            "candidate_count": len(candidates),
            "candidates": candidates,
            "semantic_trigger": raw["semantic_trigger"],
            "tracking_interpretation": "DEVELOPMENT_ONLY_UNKNOWN_TIME_GAPS",
            "metric_conversion": "NOT_PERFORMED",
            "planner_injection": "NOT_PERFORMED",
        })
    return observations


def annotate_replay_frame(frame, observation):
    image = frame.copy()
    color = (0, 0, 255) if observation["scene_state"] == "GLOBAL_CHANGE" else (0, 200, 0)
    for candidate in observation["candidates"]:
        x1, y1, x2, y2 = candidate["bbox_px"]
        cv2.rectangle(image, (x1, y1), (x2 - 1, y2 - 1), color, 3)
    cv2.putText(
        image, f"{observation['replay_index']:03d} {observation['scene_state']}", (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA,
    )
    return image
