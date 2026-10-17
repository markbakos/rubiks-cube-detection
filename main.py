import argparse
import os
from pathlib import Path
import sys

import cv2
import numpy as np

from color_detection import COLORS, recognize_face
from image_processing import load_image, rectify_face
from manual_selection import WINDOW, select_face, window_open


RESULT_WINDOW = "Recognized face"


def draw_result(face, matrix):
    canvas = np.full((410, 660, 3), 28, dtype=np.uint8)
    canvas[55:355, 15:315] = face
    for text, position in (
        ("Rectified face + sample regions", (15, 30)),
        ("Recognized colors", (345, 30)),
        ("A automatic | M manual | R reset | Q / ESC quit", (15, 380)),
        ("F White   Z Green   P Red   K Blue   N Orange   S Yellow   ? Unknown", (15, 400)),
    ):
        cv2.putText(canvas, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
    for row in range(3):
        for col in range(3):
            x, y = 15 + col * 100, 55 + row * 100
            cv2.rectangle(canvas, (x, y), (x + 99, y + 99), (120, 120, 120), 1)
            cv2.rectangle(canvas, (x + 30, y + 30), (x + 69, y + 69), (255, 255, 255), 1)
            code = matrix[row][col]
            x = 345 + col * 100
            cv2.rectangle(canvas, (x, y), (x + 99, y + 99), COLORS[code], -1)
            cv2.rectangle(canvas, (x, y), (x + 99, y + 99), (28, 28, 28), 2)
            text_color = (255, 255, 255) if code in ("P", "K", "Z", "?") else (20, 20, 20)
            cv2.putText(canvas, code, (x + 33, y + 65), cv2.FONT_HERSHEY_SIMPLEX, 1.4, text_color, 2, cv2.LINE_AA)
    return canvas


def main():
    parser = argparse.ArgumentParser(
        description="Rubik's Cube single-face recognition with automatic or manual selection."
    )
    parser.add_argument("image", type=Path, help="Path to a cube photograph (JPEG or PNG).")
    parser.add_argument("--manual", action="store_true", help="Start with manual corner selection.")
    args = parser.parse_args()

    try:
        image = load_image(args.image)
        if sys.platform.startswith("linux") and not (
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        ):
            raise ValueError("No desktop display available. Run this application in a graphical desktop session.")
        manual = args.manual
        while True:
            selection = select_face(image, manual=manual)
            if selection is None:
                return 0
            corners, mode = selection
            face = rectify_face(image, corners)
            matrix = recognize_face(face)
            print(f"\nDetected face: 1\nDetection mode: {mode}\n")
            print("\n".join(" ".join(row) for row in matrix), flush=True)
            cv2.imshow(RESULT_WINDOW, draw_result(face, matrix))
            while True:
                key = cv2.waitKey(20) & 0xFF
                if key in (27, ord("q"), ord("Q")) or not (
                    window_open(WINDOW) and window_open(RESULT_WINDOW)
                ):
                    return 0
                if key in (ord("r"), ord("R"), ord("m"), ord("M"), ord("a"), ord("A")):
                    manual = mode == "Manual" if key in (ord("r"), ord("R")) else key in (ord("m"), ord("M"))
                    cv2.destroyWindow(RESULT_WINDOW)
                    break
    except (ValueError, cv2.error) as error:
        parser.exit(1, f"Error: {error}\n")
    except KeyboardInterrupt:
        return 130
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
