import unittest

import numpy as np

from color_detection import recognize_face
from image_processing import process_image
from test_recognition import make_face


class NoiseTests(unittest.TestCase):
    def test_reproducibility_clipping_and_original_preservation(self):
        image = np.full((150, 150, 3), 128, dtype=np.uint8)
        original = image.copy()
        for noise, level in (("gaussian", 255), ("salt-pepper", 0.2)):
            with self.subTest(noise=noise):
                noisy, _ = process_image(image, noise, level)
                repeated, _ = process_image(image, noise, level)
                different, _ = process_image(image, noise, level, seed=1)
                np.testing.assert_array_equal(noisy, repeated)
                np.testing.assert_array_equal(image, original)
                self.assertFalse(np.array_equal(noisy, different))
                self.assertEqual(noisy.dtype, np.uint8)
                self.assertEqual(noisy.min(), 0)
                self.assertEqual(noisy.max(), 255)

    def test_salt_pepper_changes_whole_pixels_at_requested_density(self):
        image = np.full((200, 200, 3), 128, dtype=np.uint8)
        noisy, _ = process_image(image, "salt-pepper", 0.1)
        changed = np.any(noisy != image, axis=2)
        self.assertAlmostEqual(float(changed.mean()), 0.1, delta=0.01)
        self.assertTrue(np.all(noisy[:, :, 0] == noisy[:, :, 1]))
        self.assertTrue(np.isin(noisy, [0, 128, 255]).all())

    def test_zero_noise_and_unfiltered_output_are_independent_copies(self):
        image = np.full((20, 20, 3), 128, dtype=np.uint8)
        for noise in ("none", "gaussian", "salt-pepper"):
            noisy, processed = process_image(image, noise, 0)
            np.testing.assert_array_equal(noisy, image)
            np.testing.assert_array_equal(processed, image)
            self.assertFalse(np.shares_memory(image, noisy))
            self.assertFalse(np.shares_memory(noisy, processed))

    def test_filters_reduce_the_corresponding_noise(self):
        image = np.full((200, 200, 3), 128, dtype=np.uint8)
        for noise, level, method in (("gaussian", 30, "gaussian"), ("salt-pepper", 0.05, "median")):
            with self.subTest(noise=noise):
                noisy, processed = process_image(image, noise, level, method)
                raw_error = np.abs(noisy.astype(float) - image).mean()
                filtered_error = np.abs(processed.astype(float) - image).mean()
                self.assertLess(filtered_error, raw_error / 2)

    def test_recognition_survives_noise_and_moderate_lighting_changes(self):
        image, expected = make_face()
        for brightness in (0.65, 1.0, 1.15):
            source = np.clip(image.astype(float) * brightness, 0, 255).astype(np.uint8)
            for noise, level, method in (("gaussian", 30, "gaussian"), ("salt-pepper", 0.05, "median")):
                with self.subTest(brightness=brightness, noise=noise):
                    _, processed = process_image(source, noise, level, method)
                    self.assertEqual(recognize_face(processed), expected)

    def test_invalid_settings_are_rejected(self):
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        for settings in (
            {"noise": "invalid"}, {"noise": "none", "level": 1},
            {"noise": "gaussian", "level": -1}, {"noise": "gaussian", "level": float("nan")},
            {"noise": "gaussian", "level": float("inf")}, {"noise": "salt-pepper", "level": 1.1},
            {"filter_method": "invalid"}, {"seed": -1},
        ):
            with self.subTest(settings=settings):
                with self.assertRaises(ValueError):
                    process_image(image, **settings)


if __name__ == "__main__":
    unittest.main()
