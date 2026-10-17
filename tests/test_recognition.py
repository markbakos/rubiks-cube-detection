from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from color_detection import COLORS, classify_color, recognize_face
from image_processing import load_image, rectify_face, validate_corners


def make_face():
    matrix = [["S", "Z", "K"], ["N", "P", "F"], ["Z", "S", "N"]]
    face = np.zeros((300, 300, 3), dtype=np.uint8)
    for row in range(3):
        for col in range(3):
            face[row * 100 + 8:row * 100 + 92, col * 100 + 8:col * 100 + 92] = COLORS[matrix[row][col]]
    return face, matrix


class ImageLoadingTests(unittest.TestCase):
    def test_alpha_is_composited_on_white(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "transparent.png"
            image = np.array([[[0, 0, 0, 0], [0, 0, 255, 255], [0, 0, 255, 128]]], dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(path), image))
            np.testing.assert_array_equal(
                load_image(path), [[[255, 255, 255], [0, 0, 255], [127, 127, 255]]]
            )

    def test_grayscale_and_16_bit_input(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gray.png"
            self.assertTrue(cv2.imwrite(str(path), np.array([[0, 65535]], dtype=np.uint16)))
            np.testing.assert_array_equal(load_image(path), [[[0, 0, 0], [255, 255, 255]]])

    def test_missing_and_corrupt_images(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.png"
            with self.assertRaisesRegex(ValueError, "does not exist"):
                load_image(path)
            path.write_bytes(b"invalid image")
            with self.assertRaisesRegex(ValueError, "Could not decode"):
                load_image(path)


class RecognitionTests(unittest.TestCase):
    def test_perspective_round_trip(self):
        face, expected = make_face()
        square = np.float32([[0, 0], [299, 0], [299, 299], [0, 299]])
        corners = np.float32([[100, 60], [450, 100], [390, 420], [50, 350]])
        projected = cv2.warpPerspective(
            face, cv2.getPerspectiveTransform(square, corners), (500, 450)
        )
        original = projected.copy()
        corrected = rectify_face(projected, corners)
        self.assertEqual(corrected.shape, (300, 300, 3))
        self.assertEqual(recognize_face(corrected), expected)
        np.testing.assert_array_equal(projected, original)

    def test_center_sampling_resists_borders_and_isolated_outliers(self):
        face, expected = make_face()
        face[::5, ::5] = (255, 255, 255)
        self.assertEqual(recognize_face(face), expected)

    def test_red_wrap_white_and_ambiguous_samples(self):
        for hsv, expected in (
            ((0, 255, 255), "P"),
            ((179, 255, 255), "P"),
            ((0, 0, 120), "F"),
            ((0, 0, 30), "?"),
            ((60, 255, 30), "?"),
            ((150, 255, 255), "?"),
        ):
            with self.subTest(hsv=hsv):
                self.assertEqual(classify_color(hsv), expected)

    def test_invalid_corner_selections(self):
        face, _ = make_face()
        for corners in (
            [],
            [[0, 0], [299, 0], [299, 299]],
            [[0, 0], [299, 299], [299, 0], [0, 299]],
            [[0, 0], [299, 0], [100, 20], [299, 299]],
            [[0, 0], [0, 0], [299, 299], [0, 299]],
            [[0, 0], [299, 0], [300, 299], [0, 299]],
            [[0, 0], [299, 0], [float("nan"), 299], [0, 299]],
            [[0, 0], [0, 299], [299, 299], [299, 0]],
            [[0, 0], [1, 0], [1, 1], [0, 1]],
        ):
            with self.subTest(corners=corners):
                with self.assertRaises(ValueError):
                    validate_corners(face, corners)

    def test_wrong_face_dimensions_are_rejected(self):
        with self.assertRaises(ValueError):
            recognize_face(np.zeros((100, 100, 3), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
