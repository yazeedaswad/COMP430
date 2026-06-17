from __future__ import annotations

import cv2
import numpy as np

from pill_counting.models.instance_separation import InstanceSeparationResult, clean_foreground


def center_marker_watershed(
    foreground_prob: np.ndarray,
    center_prob: np.ndarray,
    foreground_threshold: float = 0.5,
    center_threshold: float = 0.35,
    peak_min_distance: int = 24,
    min_area: int = 20,
    max_area_ratio: float = 0.0,
    min_solidity: float = 0.0,
    max_aspect_ratio: float = 0.0,
    kernel_size: int = 3,
) -> InstanceSeparationResult:
    foreground = (foreground_prob >= foreground_threshold).astype(np.uint8)
    foreground = clean_foreground(foreground, kernel_size=kernel_size)
    if foreground.sum() == 0:
        empty = np.zeros_like(foreground, dtype=np.int32)
        return InstanceSeparationResult(count=0, labels=empty, foreground=foreground, markers=empty)

    center = center_prob.copy().astype(np.float32)
    center[foreground == 0] = 0
    local_max = center == cv2.dilate(center, np.ones((7, 7), np.uint8))
    ys, xs = np.where((center >= center_threshold) & local_max & (foreground > 0))
    candidates = sorted(
        [(float(center[y, x]), int(x), int(y)) for y, x in zip(ys, xs)],
        reverse=True,
    )
    selected: list[tuple[int, int]] = []
    for _, x, y in candidates:
        if all((x - sx) ** 2 + (y - sy) ** 2 >= peak_min_distance**2 for sx, sy in selected):
            selected.append((x, y))

    marker_mask = np.zeros_like(foreground, dtype=np.uint8)
    for x, y in selected:
        cv2.circle(marker_mask, (x, y), 2, 1, thickness=-1)

    num_markers, markers = cv2.connectedComponents((marker_mask > 0).astype(np.uint8))
    if num_markers <= 1:
        empty = np.zeros_like(foreground, dtype=np.int32)
        return InstanceSeparationResult(count=0, labels=empty, foreground=foreground, markers=empty)

    watershed_input = cv2.cvtColor((foreground * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    watershed_markers = markers.astype(np.int32)
    watershed_markers[foreground == 0] = 0
    cv2.watershed(watershed_input, watershed_markers)

    output = np.zeros_like(watershed_markers, dtype=np.int32)
    count = 0
    image_area = foreground.shape[0] * foreground.shape[1]
    for marker_id in sorted(set(np.unique(watershed_markers)) - {-1, 0}):
        component = (watershed_markers == marker_id).astype(np.uint8)
        area = int(component.sum())
        if area < min_area:
            continue
        if max_area_ratio > 0 and area > image_area * max_area_ratio:
            continue

        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = max(w / max(h, 1), h / max(w, 1))
        if max_aspect_ratio > 0 and aspect_ratio > max_aspect_ratio:
            continue
        hull = cv2.convexHull(contour)
        hull_area = max(float(cv2.contourArea(hull)), 1.0)
        solidity = float(cv2.contourArea(contour)) / hull_area
        if min_solidity > 0 and solidity < min_solidity:
            continue

        count += 1
        output[component > 0] = count

    return InstanceSeparationResult(count=count, labels=output, foreground=foreground, markers=markers)
