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
    def run_selection(self, image, actions, manual=True, candidates=None, settings=None):
        callbacks = []

        def keypress(delay):
            clicks, key = next(actions, ([], 27))
            for x, y in clicks:
                callbacks[0](cv2.EVENT_LBUTTONDOWN, x, y + manual_selection.HEADER_HEIGHT, 0, None)
            return key

        with (
            patch.object(cv2, "namedWindow"),
            patch.object(cv2, "imshow"),
            patch.object(cv2, "setMouseCallback", side_effect=lambda name, callback: callbacks.append(callback)),
            patch.object(cv2, "getWindowProperty", return_value=1),
            patch.object(cv2, "waitKey", side_effect=keypress),
            patch.object(manual_selection, "find_face_candidates", return_value=candidates or []),
        ):
            return manual_selection.select_face(image, manual=manual, settings=settings)

    def test_noise_changes_preserve_manual_points_and_restore_the_original(self):
        image, _ = make_face()
        original = image.copy()
        settings = {"noise": "none", "level": 0, "filter_method": "none", "seed": 7}
        actions = iter([
            ([(10, 10), (290, 10), (290, 290), (10, 290)], ord("g")),
            ([], ord("p")), ([], ord("f")), ([], ord("o")), ([], 13),
        ])
        corners, mode = self.run_selection(image, actions, settings=settings)
        self.assertEqual(mode, "Manual")
        self.assertEqual(settings, {"noise": "none", "level": 0, "filter_method": "none", "seed": 7})
        np.testing.assert_array_equal(corners, [[10, 10], [290, 10], [290, 290], [10, 290]])
        np.testing.assert_array_equal(image, original)

    def test_noise_and_filter_presets_cycle(self):
        settings = {"noise": "none", "level": 0, "filter_method": "none", "seed": 0}
        for key, noise, level in (("G", "gaussian", 5), ("g", "gaussian", 15), ("g", "gaussian", 30), ("g", "gaussian", 5), ("p", "salt-pepper", 0.005), ("p", "salt-pepper", 0.02), ("p", "salt-pepper", 0.05), ("p", "salt-pepper", 0.005)):
            self.assertTrue(manual_selection.change_experiment(settings, ord(key)))
            self.assertEqual((settings["noise"], settings["level"]), (noise, level))
        for method in ("gaussian", "median", "none"):
            self.assertTrue(manual_selection.change_experiment(settings, ord("f")))
            self.assertEqual(settings["filter_method"], method)
        self.assertFalse(manual_selection.change_experiment(settings, ord("r")))

    def test_resized_click_coordinates_and_ignored_clicks(self):
        image = np.zeros((1600, 2000, 3), dtype=np.uint8)
        actions = iter([
            ([(0, -1), (800, 100), (10, 10), (650, 10), (650, 500), (10, 500), (20, 100)], 13),
        ])
        corners, mode = self.run_selection(image, actions)
        np.testing.assert_allclose(
            corners,
            np.array([[10, 10], [650, 10], [650, 500], [10, 500]]) * [2000 / 688, 1600 / 550],
            rtol=1e-6,
        )

    def test_incomplete_invalid_selection_and_reset(self):
        image = np.zeros((300, 300, 3), dtype=np.uint8)
        actions = iter([
            ([(10, 10), (290, 290)], 13),
            ([(290, 10), (10, 290)], 13),
            ([], ord("r")),
            ([(10, 10), (290, 10), (290, 290), (10, 290)], 13),
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
                    actions.append(([(10, 10), (290, 10), (290, 290), (10, 290)], 13))
                else:
                    actions.append(([], 13))
                corners, mode = self.run_selection(image, iter(actions), manual=False, candidates=[automatic])
                self.assertEqual(mode, "Manual" if override else "Automatic")
                expected = [[10, 10], [290, 10], [290, 290], [10, 290]] if override else automatic
                np.testing.assert_array_equal(corners, expected)

    def test_failed_automatic_detection_falls_back_to_manual(self):
        image = np.zeros((300, 300, 3), dtype=np.uint8)
        actions = iter([([(10, 10), (290, 10), (290, 290), (10, 290)], 13)])
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

    def test_reset_preserves_automatic_mode_after_manual_switches(self):
        image = np.zeros((300, 300, 3), dtype=np.uint8)
        automatic = np.float32([[0, 0], [299, 0], [299, 299], [0, 299]])
        for keys in ("mrar", "MRAR"):
            with self.subTest(keys=keys):
                actions = iter([([], ord(key)) for key in keys] + [([], 13)])
                selection = self.run_selection(image, actions, manual=False, candidates=[automatic])
                self.assertIsNotNone(selection)
                corners, mode = selection
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
    def test_noise_changes_in_manual_results_reuse_corners_without_accumulating_noise(self):
        face, _ = make_face()
        original = face.copy()
        corners = np.float32([[0, 0], [299, 0], [299, 299], [0, 299]])
        with (
            patch("sys.argv", ["main.py", "cube.png", "--manual"]),
            patch.dict("os.environ", {"DISPLAY": ":test"}),
            patch.object(main, "load_image", return_value=face),
            patch.object(main, "select_face", return_value=(corners, "Manual")) as selection,
            patch.object(main, "show_recognition") as recognition,
            patch.object(main, "window_open", return_value=True),
            patch.object(cv2, "imshow"),
            patch.object(cv2, "waitKey", side_effect=[ord(key) for key in "gpfoq"]),
            patch.object(cv2, "destroyAllWindows"),
        ):
            self.assertEqual(main.main(), 0)
        selection.assert_called_once()
        self.assertEqual(recognition.call_count, 5)
        np.testing.assert_array_equal(recognition.call_args.args[0], original)
        np.testing.assert_array_equal(face, original)

    def test_noise_changes_in_automatic_results_request_new_detection(self):
        face, _ = make_face()
        corners = np.float32([[0, 0], [299, 0], [299, 299], [0, 299]])
        with (
            patch("sys.argv", ["main.py", "cube.png"]),
            patch.dict("os.environ", {"DISPLAY": ":test"}),
            patch.object(main, "load_image", return_value=face),
            patch.object(main, "select_face", side_effect=[(corners, "Automatic"), None]) as selection,
            patch.object(main, "window_open", return_value=True),
            patch.object(cv2, "imshow"),
            patch.object(cv2, "waitKey", return_value=ord("g")),
            patch.object(cv2, "destroyWindow"),
            patch.object(cv2, "destroyAllWindows"),
            redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(main.main(), 0)
        self.assertFalse(selection.call_args.kwargs["manual"])
        self.assertEqual(selection.call_args.kwargs["settings"]["noise"], "gaussian")
        self.assertEqual(selection.call_args.kwargs["settings"]["level"], 5)

    def test_result_reset_preserves_mode_after_switching_back_to_automatic(self):
        face, _ = make_face()
        corners = np.float32([[0, 0], [299, 0], [299, 299], [0, 299]])
        selections = [(corners, "Automatic"), (corners, "Manual"), (corners, "Automatic"), None]
        with (
            patch("sys.argv", ["main.py", "cube.png"]),
            patch.dict("os.environ", {"DISPLAY": ":test"}),
            patch.object(main, "load_image", return_value=face),
            patch.object(main, "select_face", side_effect=selections) as selection,
            patch.object(main, "window_open", return_value=True),
            patch.object(cv2, "imshow"),
            patch.object(cv2, "waitKey", side_effect=[ord("m"), ord("a"), ord("r")]),
            patch.object(cv2, "destroyWindow"),
            patch.object(cv2, "destroyAllWindows"),
            redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(main.main(), 0)
        self.assertEqual([call.kwargs["manual"] for call in selection.call_args_list], [False, True, False, False])

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
            self.assertEqual(display.call_args.args[1].shape, (435, 660, 3))
        self.assertIn("Detection mode: Manual", output.getvalue())
        self.assertIn("\n".join(" ".join(row) for row in expected), output.getvalue())


if __name__ == "__main__":
    unittest.main()
