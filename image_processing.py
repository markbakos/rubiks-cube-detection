from pathlib import Path

import cv2
import numpy as np


def load_image(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Image does not exist: {path}")
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not decode image: {path}")
    if image.dtype == np.uint16:
        image = np.round(image / 257).astype(np.uint8)
    if image.dtype != np.uint8:
        raise ValueError("Only 8-bit and 16-bit images are supported.")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        alpha = image[:, :, 3:4] / 255.0
        return np.round(image[:, :, :3] * alpha + 255 * (1 - alpha)).astype(np.uint8)
    if image.shape[2] != 3:
        raise ValueError("The image must have grayscale, BGR, or BGRA channels.")
    return image


def validate_corners(image, corners):
    corners = np.asarray(corners, dtype=np.float32)
    if corners.shape != (4, 2) or not np.isfinite(corners).all():
        raise ValueError("Select exactly four finite corner points.")
    height, width = image.shape[:2]
    if (
        (corners < 0).any()
        or (corners[:, 0] > width - 1).any()
        or (corners[:, 1] > height - 1).any()
    ):
        raise ValueError("All corners must lie inside the image.")
    if not cv2.isContourConvex(corners) or cv2.contourArea(corners) < 25:
        raise ValueError("Select a convex face with four distinct corners.")
    if cv2.contourArea(corners, oriented=True) < 0:
        raise ValueError("Select corners clockwise: TL, TR, BR, BL.")
    return corners


def rectify_face(image, corners):
    corners = validate_corners(image, corners)
    destination = np.float32([[0, 0], [299, 0], [299, 299], [0, 299]])
    transform = cv2.getPerspectiveTransform(corners, destination)
    return cv2.warpPerspective(image, transform, (300, 300))


def process_image(image, noise="none", level=0, filter_method="none", seed=0):
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        raise ValueError("Expected a nonempty 8-bit BGR image.")
    if noise not in ("none", "gaussian", "salt-pepper"):
        raise ValueError("Unknown noise type.")
    maximum = 1 if noise == "salt-pepper" else 255 if noise == "gaussian" else 0
    if not np.isfinite(level) or not 0 <= level <= maximum:
        raise ValueError(f"Noise level for {noise} must be between 0 and {maximum}.")
    if filter_method not in ("none", "gaussian", "median"):
        raise ValueError("Unknown filter type.")
    rng = np.random.default_rng(seed)
    noisy = image.copy()
    if noise == "gaussian" and level:
        noisy = np.clip(np.rint(image.astype(np.float32) + rng.normal(0, level, image.shape)), 0, 255).astype(np.uint8)
    elif noise == "salt-pepper" and level:
        mask = rng.random(image.shape[:2])
        noisy[mask < level / 2] = 0
        noisy[mask >= 1 - level / 2] = 255
    if filter_method == "gaussian":
        processed = cv2.GaussianBlur(noisy, (5, 5), 0)
    elif filter_method == "median":
        processed = cv2.medianBlur(noisy, 3)
    else:
        processed = noisy.copy()
    return noisy, processed
