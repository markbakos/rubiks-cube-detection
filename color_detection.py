import cv2
import numpy as np


COLORS = {
    "F": (240, 240, 240),
    "Z": (45, 160, 45),
    "P": (40, 40, 220),
    "K": (210, 90, 30),
    "N": (0, 140, 255),
    "S": (0, 225, 255),
    "?": (100, 100, 100),
}


def classify_color(hsv):
    hue, saturation, value = hsv
    if value < 45:
        return "?"
    if saturation < 60:
        return "F" if value >= 90 else "?"
    if hue <= 8 or hue >= 170:
        return "P"
    if hue <= 22:
        return "N"
    if hue <= 38:
        return "S"
    if hue <= 89:
        return "Z"
    if hue <= 134:
        return "K"
    return "?"


def recognize_face(face):
    if face.shape != (300, 300, 3) or face.dtype != np.uint8:
        raise ValueError("Expected a 300x300, 8-bit BGR face image.")
    samples = np.array(
        [
            [
                np.median(
                    face[row * 100 + 30:row * 100 + 70, col * 100 + 30:col * 100 + 70],
                    axis=(0, 1),
                )
                for col in range(3)
            ]
            for row in range(3)
        ],
        dtype=np.uint8,
    )
    hsv_samples = cv2.cvtColor(samples, cv2.COLOR_BGR2HSV)
    return [[classify_color(sample) for sample in row] for row in hsv_samples]
