from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import evaluation
from test_recognition import make_face


class EvaluationTests(unittest.TestCase):
    def test_rotated_detection_is_aligned_to_the_reference(self):
        image, matrix = make_face()
        corners = np.float32([[0, 0], [299, 0], [299, 299], [0, 299]])
        with patch.object(evaluation, "find_face_candidates", return_value=[np.roll(corners, 1, axis=0)]):
            result = evaluation.evaluate_image(image, [(corners, np.asarray(matrix))])
        self.assertEqual(result["detection"], "matched")
        self.assertEqual(result["reference_correct"], 9)
        self.assertEqual(result["automatic_correct"], 9)
        self.assertEqual(result["automatic_face_correct"], 1)

    def test_missing_and_unannotated_detections_have_no_invented_accuracy(self):
        image, matrix = make_face()
        corners = np.float32([[0, 0], [299, 0], [299, 299], [0, 299]])
        for candidates, status in (([], "none"), ([corners / 3], "unmatched")):
            with self.subTest(status=status):
                with patch.object(evaluation, "find_face_candidates", return_value=candidates):
                    result = evaluation.evaluate_image(image, [(corners, np.asarray(matrix))])
                self.assertEqual(result["detection"], status)
                self.assertEqual(result["automatic_correct"], "")
                self.assertEqual(result["reference_correct"], 9)

    def test_unknown_cells_are_counted_as_errors(self):
        image, matrix = make_face()
        image[:] = 0
        corners = np.float32([[0, 0], [299, 0], [299, 299], [0, 299]])
        with patch.object(evaluation, "find_face_candidates", return_value=[corners]):
            result = evaluation.evaluate_image(image, [(corners, np.asarray(matrix))])
        self.assertEqual(result["reference_correct"], 0)
        self.assertEqual(result["reference_unknown"], 9)
        self.assertEqual(result["automatic_unknown"], 9)

    def test_annotation_validation_and_report_export(self):
        image, matrix = make_face()
        corners = [[0, 0], [299, 0], [299, 299], [0, 299]]
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            self.assertTrue(cv2.imwrite(str(directory / "cube.png"), image))
            path = directory / "annotations.json"
            record = [{"image": "cube.png", "faces": [{"corners": corners, "matrix": matrix}]}]
            path.write_text(json.dumps(record))
            output = directory / "reports/results.csv"
            with (
                patch("sys.argv", ["evaluation.py", str(path), "--output", str(output)]),
                patch.object(evaluation, "find_face_candidates", return_value=[]),
                redirect_stdout(io.StringIO()),
            ):
                evaluation.main()
            with output.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 13)
            self.assertTrue(all(row["detection"] == "none" for row in rows))
            self.assertEqual(rows[0]["reference_correct"], "9")
            for invalid in ([], [{"image": "cube.png", "faces": []}], [{"image": "cube.png", "faces": [{"corners": corners, "matrix": [["?"] * 3] * 3}]}]):
                path.write_text(json.dumps(invalid))
                with self.assertRaises(ValueError):
                    evaluation.load_annotations(path)


if __name__ == "__main__":
    unittest.main()
