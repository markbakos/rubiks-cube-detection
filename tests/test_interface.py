from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import main
import manual_selection
from test_recognition import make_face


class ManualSelectionTests(unittest.TestCase):
    def run_selection(self, image, actions, manual=True, candidates=None):
        callbacks = []

        def keypress(delay):
            clicks, key = next(actions)
            for x, y in clicks:
                callbacks[0](cv2.EVENT_LBUTTONDOWN, x, y, 0, None)
            return key

        with (
            patch.object(cv2, "namedWindow"),
            patch.object(cv2, "imshow"),
            patch.object(cv2, "setMouseCallback", side_effect=lambda name, callback: callbacks.append(callback)),
            patch.object(cv2, "getWindowProperty", return_value=1),
            patch.object(cv2, "waitKey", side_effect=keypress),
            patch.object(manual_selection, "find_face_candidates", return_value=candidates or []),
        ):
            return manual_selection.select_face(image, manual=manual)

    def test_resized_click_coordinates_and_ignored_clicks(self):
        image = np.zeros((1600, 2000, 3), dtype=np.uint8)
        actions = iter([
            ([(0, 0), (800, 200), (10, 110), (700, 110), (700, 700), (10, 700), (20, 200)], 13),
        ])
        corners, mode = self.run_selection(image, actions)
        np.testing.assert_allclose(
            corners,
            np.array([[10, 10], [700, 10], [700, 600], [10, 600]]) * [2000 / 775, 1600 / 620],
            rtol=1e-6,
        )

    def test_incomplete_invalid_selection_and_reset(self):
        image = np.zeros((300, 300, 3), dtype=np.uint8)
        actions = iter([
            ([(10, 110), (290, 390)], 13),
            ([(290, 110), (10, 390)], 13),
            ([], ord("r")),
            ([(10, 110), (290, 110), (290, 390), (10, 390)], 13),
        ])
        corners, mode = self.run_selection(image, actions)
        np.testing.assert_array_equal(corners, [[10, 10], [290, 10], [290, 290], [10, 290]])

    def test_automatic_preview_can_be_accepted_or_replaced_manually(self):
        image = np.zeros((300, 300, 3), dtype=np.uint8)
        automatic = np.float32([[0, 0], [299, 0], [299, 299], [0, 299]])
        for override in (False, True):
            with self.subTest(override=override):
                actions = []
                if override:
                    actions.append(([], ord("m")))
                    actions.append(([(10, 110), (290, 110), (290, 390), (10, 390)], 13))
                else:
                    actions.append(([], 13))
                corners, mode = self.run_selection(image, iter(actions), manual=False, candidates=[automatic])
                self.assertEqual(mode, "Manual" if override else "Automatic")
                expected = [[10, 10], [290, 10], [290, 290], [10, 290]] if override else automatic
                np.testing.assert_array_equal(corners, expected)

    def test_failed_automatic_detection_falls_back_to_manual(self):
        image = np.zeros((300, 300, 3), dtype=np.uint8)
        actions = iter([([(10, 110), (290, 110), (290, 390), (10, 390)], 13)])
        corners, mode = self.run_selection(image, actions, manual=False)
        self.assertEqual(mode, "Manual")
        np.testing.assert_array_equal(corners, [[10, 10], [290, 10], [290, 290], [10, 290]])

    def test_automatic_selection_can_be_requested_from_manual_mode(self):
        image = np.zeros((300, 300, 3), dtype=np.uint8)
        automatic = np.float32([[0, 0], [299, 0], [299, 299], [0, 299]])
        actions = iter([([], ord("a")), ([], 13)])
        corners, mode = self.run_selection(image, actions, candidates=[automatic])
        self.assertEqual(mode, "Automatic")
        np.testing.assert_array_equal(corners, automatic)

    def test_escape_cancels(self):
        self.assertIsNone(self.run_selection(np.zeros((300, 300, 3), dtype=np.uint8), iter([([], 27)])))

    def test_closed_window_cancels(self):
        with (
            patch.object(cv2, "namedWindow"),
            patch.object(cv2, "imshow"),
            patch.object(cv2, "setMouseCallback"),
            patch.object(cv2, "waitKey", return_value=-1),
            patch.object(cv2, "getWindowProperty", side_effect=cv2.error("window closed")),
        ):
            self.assertIsNone(manual_selection.select_face(np.zeros((300, 300, 3), dtype=np.uint8), manual=True))


class ApplicationTests(unittest.TestCase):
    def test_manual_pipeline_prints_matrix_and_allows_reselection(self):
        face, expected = make_face()
        corners = np.float32([[0, 0], [299, 0], [299, 299], [0, 299]])
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cube.png"
            self.assertTrue(cv2.imwrite(str(path), face))
            with (
                patch("sys.argv", ["main.py", str(path), "--manual"]),
                patch.dict("os.environ", {"DISPLAY": ":test"}),
                patch.object(main, "select_face", side_effect=[(corners, "Manual"), None]) as selection,
                patch.object(main, "window_open", return_value=True),
                patch.object(cv2, "imshow") as display,
                patch.object(cv2, "waitKey", return_value=ord("r")),
                patch.object(cv2, "destroyWindow") as close_result,
                patch.object(cv2, "destroyAllWindows") as close_all,
                redirect_stdout(output),
            ):
                self.assertEqual(main.main(), 0)
            self.assertEqual(selection.call_count, 2)
            self.assertTrue(selection.call_args_list[0].kwargs["manual"])
            self.assertTrue(selection.call_args.kwargs["manual"])
            close_result.assert_called_once_with(main.RESULT_WINDOW)
            close_all.assert_called_once()
            self.assertEqual(display.call_args.args[1].shape, (410, 660, 3))
        self.assertIn("Detection mode: Manual", output.getvalue())
        self.assertIn("\n".join(" ".join(row) for row in expected), output.getvalue())


if __name__ == "__main__":
    unittest.main()
