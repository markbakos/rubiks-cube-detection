from pathlib import Path
import unittest

import cv2
import numpy as np

from color_detection import recognize_face
from face_detection import find_face_candidates
from image_processing import load_image, rectify_face, validate_corners
from test_recognition import make_face


def overlap(first, second):
    intersection = cv2.intersectConvexConvex(first, second)[0]
    return intersection / (cv2.contourArea(first) + cv2.contourArea(second) - intersection)


class FaceDetectionTests(unittest.TestCase):
    def test_perspective_face_and_original_coordinate_scaling(self):
        face, expected = make_face()
        square = np.float32([[0, 0], [299, 0], [299, 299], [0, 299]])
        reference = np.float32([[100, 60], [450, 100], [390, 420], [50, 350]])
        projected = cv2.warpPerspective(
            face, cv2.getPerspectiveTransform(square, reference), (500, 450), borderValue=(220, 220, 220)
        )
        for factor in (1, 3):
            with self.subTest(factor=factor):
                image = cv2.resize(projected, None, fx=factor, fy=factor)
                original = image.copy()
                candidates = find_face_candidates(image)
                self.assertTrue(candidates)
                corners = candidates[0]
                validate_corners(image, corners)
                self.assertGreater(overlap(corners, reference * factor), 0.8)
                self.assertEqual(recognize_face(rectify_face(image, corners)), expected)
                np.testing.assert_array_equal(image, original)

    def test_blank_single_sticker_and_wrong_grid_are_rejected(self):
        blank = np.full((400, 400, 3), 240, dtype=np.uint8)
        sticker = blank.copy()
        cv2.rectangle(sticker, (50, 50), (350, 350), (0, 0, 0), -1)
        cv2.rectangle(sticker, (60, 60), (340, 340), (0, 180, 255), -1)
        wrong_grid = blank.copy()
        cv2.rectangle(wrong_grid, (40, 40), (360, 360), (0, 0, 0), -1)
        for row in range(2):
            for col in range(2):
                start = (50 + col * 160, 50 + row * 160)
                cv2.rectangle(wrong_grid, start, (start[0] + 140, start[1] + 140), (0, 180, 255), -1)
        for image in (blank, sticker, wrong_grid):
            with self.subTest():
                self.assertEqual(find_face_candidates(image), [])

    def test_random_texture_is_rejected(self):
        image = np.random.default_rng(4).integers(0, 256, (160, 200, 3), dtype=np.uint8)
        self.assertEqual(find_face_candidates(image), [])

    def test_supplied_photographs_include_a_complete_verified_face(self):
        references = {
            "Rubik01.jpg": (
                [[323, 78], [669, 215], [434, 537], [85, 360]],
                [["P", "Z", "P"], ["K", "S", "F"], ["P", "F", "P"]],
            ),
            "Rubik02.jpg": (
                [[189, 94], [392, 0], [590, 94], [391, 212]],
                [["P", "F", "P"], ["F", "K", "F"], ["K", "F", "F"]],
            ),
            "Rubik03.jpg": (
                [[132, 220], [303, 95], [562, 130], [460, 289]],
                [["K", "S", "Z"], ["Z", "P", "P"], ["Z", "Z", "Z"]],
            ),
            "Rubik04.png": (
                [[80, 176], [414, 62], [725, 178], [397, 315]],
                [["K", "F", "Z"], ["F", "S", "P"], ["P", "K", "N"]],
            ),
        }
        directory = Path(__file__).resolve().parents[1] / "images"
        for filename, (corners, expected) in references.items():
            with self.subTest(image=filename):
                image = load_image(directory / filename)
                reference = np.float32(corners)
                candidates = find_face_candidates(image)
                self.assertTrue(candidates)
                match = max(candidates, key=lambda candidate: overlap(candidate, reference))
                self.assertGreater(overlap(match, reference), 0.85)
                self.assertEqual(recognize_face(rectify_face(image, match)), expected)


if __name__ == "__main__":
    unittest.main()
