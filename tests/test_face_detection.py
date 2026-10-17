from pathlib import Path
import unittest

import cv2
import numpy as np

from color_detection import recognize_face
from evaluation import load_annotations
from face_detection import _merge_lines, find_face_candidates
from image_processing import rectify_face, validate_corners
from test_recognition import make_face


def overlap(first, second):
    intersection = cv2.intersectConvexConvex(first, second)[0]
    return intersection / (cv2.contourArea(first) + cv2.contourArea(second) - intersection)


class FaceDetectionTests(unittest.TestCase):
    def test_collinear_segments_merge_without_merging_separate_grid_lines(self):
        segments = np.array([[0, 0, 100, 0], [90, 1, 200, 1], [0, 30, 150, 30], [0, 60, 50, 60]])
        merged = _merge_lines(segments, 2)
        np.testing.assert_allclose(merged, [[[0, 1], [200, 1]], [[0, 30], [150, 30]], [[0, 60], [50, 60]]])

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
        directory = Path(__file__).resolve().parents[1] / "images"
        for filename, image, faces in load_annotations(directory / "annotations.json"):
            with self.subTest(image=filename):
                reference, expected = faces[0]
                candidates = find_face_candidates(image)
                self.assertTrue(candidates)
                match = max(candidates, key=lambda candidate: overlap(candidate, reference))
                self.assertGreater(overlap(match, reference), 0.85)
                self.assertEqual(recognize_face(rectify_face(image, match)), expected.tolist())


if __name__ == "__main__":
    unittest.main()
