from itertools import combinations

import cv2
import numpy as np


def _merge_lines(segments, tolerance):
    merged = []
    lengths = np.linalg.norm(segments[:, 2:] - segments[:, :2], axis=1)
    for segment in segments[np.argsort(-lengths)]:
        endpoints = segment.reshape(2, 2).astype(np.float64)
        for existing in merged:
            direction = existing[1] - existing[0]
            direction /= np.linalg.norm(direction)
            normal = np.array([-direction[1], direction[0]])
            distances = (endpoints - existing[0]) @ normal
            if np.max(np.abs(distances)) <= tolerance:
                positions = (endpoints - existing[0]) @ direction
                length = np.linalg.norm(existing[1] - existing[0])
                start = existing[0].copy()
                existing[0] = start + min(0, positions.min()) * direction
                existing[1] = start + max(length, positions.max()) * direction
                break
        else:
            merged.append(endpoints)
    merged.sort(key=lambda line: np.linalg.norm(line[1] - line[0]), reverse=True)
    return np.array(merged[:40])


def _grid_score(edges, corners):
    destination = np.float32([[0, 0], [149, 0], [149, 149], [0, 149]])
    transform = cv2.getPerspectiveTransform(corners, destination)
    warped = cv2.warpPerspective(edges, transform, (150, 150), flags=cv2.INTER_LINEAR)
    interiors = [
        np.mean(warped[row * 50 + 15:row * 50 + 35, col * 50 + 15:col * 50 + 35] > 0)
        for row in range(3)
        for col in range(3)
    ]
    if np.mean(interiors) > 0.12:
        return 0
    support = []
    for view in (warped, warped.T):
        for division in (50, 100):
            strip = view[:, division - 7:division + 8].max(axis=1) > 0
            for start, end in ((8, 42), (58, 92), (108, 142)):
                support.append(np.mean(strip[start:end]))
    if min(support) < 0.35 or np.mean(support) < 0.65:
        return 0
    return float(0.5 * min(support) + 0.5 * np.mean(support))


def find_face_candidates(image):
    height, width = image.shape[:2]
    scale = min(1, 800 / max(height, width))
    resized = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))))
    gray = cv2.GaussianBlur(cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.Canny(gray, 40, 120)
    size = min(gray.shape)
    segments = cv2.HoughLinesP(
        edges, 1, np.pi / 180, 40,
        minLineLength=max(15, size * 0.08), maxLineGap=max(3, size * 0.04),
    )
    if segments is None:
        return []
    lines = _merge_lines(segments.reshape(-1, 4), max(3, size * 0.008))
    if len(lines) < 4:
        return []
    directions = lines[:, 1] - lines[:, 0]
    lengths = np.linalg.norm(directions, axis=1)
    directions /= lengths[:, None]
    normals = np.column_stack((-directions[:, 1], directions[:, 0]))
    coefficients = np.column_stack((normals, -np.sum(normals * lines[:, 0], axis=1)))
    intersections = np.cross(coefficients[:, None, :], coefficients[None, :, :])
    valid = np.abs(intersections[:, :, 2]) > 1e-6
    coordinates = np.full(intersections.shape[:2] + (2,), np.nan)
    coordinates[valid] = intersections[valid, :2] / intersections[valid, 2:3]
    pairs = [
        (a, b)
        for a, b in combinations(range(len(lines)), 2)
        if abs(directions[a] @ directions[b]) > np.cos(np.deg2rad(25))
        and abs((lines[b, 0] - lines[a, 0]) @ normals[a]) > size * 0.08
    ]
    candidates = []
    image_area = gray.size
    for (a, b), (c, d) in combinations(pairs, 2):
        if len({a, b, c, d}) != 4 or abs(directions[a] @ directions[c]) > np.cos(np.deg2rad(25)):
            continue
        corners = coordinates[[a, a, b, b], [c, d, d, c]].astype(np.float32)
        if not np.isfinite(corners).all() or (corners < -2).any():
            continue
        if (corners[:, 0] > gray.shape[1] + 1).any() or (corners[:, 1] > gray.shape[0] + 1).any():
            continue
        corners[:, 0] = np.clip(corners[:, 0], 0, gray.shape[1] - 1)
        corners[:, 1] = np.clip(corners[:, 1], 0, gray.shape[0] - 1)
        area = cv2.contourArea(corners)
        if area < image_area * 0.035 or not cv2.isContourConvex(corners):
            continue
        supported = True
        for index, line_index in enumerate((a, d, b, c)):
            edge = corners[(index + 1) % 4] - corners[index]
            edge_length = np.linalg.norm(edge)
            if edge_length < size * 0.08:
                supported = False
                break
            positions = (lines[line_index] - corners[index]) @ (edge / edge_length)
            overlap = min(edge_length, positions.max()) - max(0, positions.min())
            if overlap < edge_length * 0.5:
                supported = False
                break
        if not supported:
            continue
        if cv2.contourArea(corners, oriented=True) < 0:
            corners = corners[::-1].copy()
        corners = np.roll(corners, -np.argmin(corners.sum(axis=1)), axis=0)
        score = _grid_score(edges, corners)
        if score:
            candidates.append((score + 0.05 * np.sqrt(area / image_area), corners))
    candidates.sort(key=lambda candidate: candidate[0], reverse=True)
    selected = []
    for score, corners in candidates:
        if all(
            cv2.intersectConvexConvex(corners, existing)[0]
            / min(cv2.contourArea(corners), cv2.contourArea(existing)) < 0.8
            for existing in selected
        ):
            selected.append(corners)
    ratio = np.array([width / gray.shape[1], height / gray.shape[0]], dtype=np.float32)
    return [np.minimum(corners * ratio, [width - 1, height - 1]).astype(np.float32) for corners in selected]
