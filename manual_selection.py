import cv2
import numpy as np

from image_processing import validate_corners
from face_detection import find_face_candidates


WINDOW = "Cube face selection"
HEADER_HEIGHT = 100


def window_open(name):
    try:
        return cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        return False


def select_face(image, manual=False):
    height, width = image.shape[:2]
    scale = min(1.0, 1000 / width, 620 / height)
    display_width = max(1, round(width * scale))
    display_height = max(1, round(height * scale))
    display = cv2.resize(image, (display_width, display_height))
    canvas_width = max(660, display_width)
    points = []
    mode = "Manual"
    message = "Select the outer corners of one complete 3x3 face."

    def redraw():
        canvas = np.full((display_height + HEADER_HEIGHT, canvas_width, 3), 28, dtype=np.uint8)
        canvas[HEADER_HEIGHT:, :display_width] = display
        for y, text, color in (
            (25, "Click corners: 1 TL -> 2 TR -> 3 BR -> 4 BL" if mode == "Manual" else "Automatic face selected - ENTER accept or M select manually", (245, 245, 245)),
            (50, "ENTER recognize | A auto | M / R manual | Q / ESC quit", (190, 190, 190)),
            (78, f"{mode} {len(points)}/4  {message}", (80, 220, 255)),
        ):
            cv2.putText(canvas, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        if points:
            drawn_points = np.rint(
                np.array(points) * [display_width / width, display_height / height] + [0, HEADER_HEIGHT]
            ).astype(np.int32)
            cv2.polylines(canvas, [drawn_points], len(points) == 4, (0, 220, 255), 2, cv2.LINE_AA)
            for index, point in enumerate(drawn_points, start=1):
                x, y = point
                cv2.circle(canvas, (x, y), 5, (0, 220, 255), -1, cv2.LINE_AA)
                label = (min(x + 9, canvas_width - 20), max(HEADER_HEIGHT + 18, y - 9))
                cv2.putText(canvas, str(index), label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(canvas, str(index), label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(WINDOW, canvas)

    def on_mouse(event, x, y, flags, param):
        nonlocal message
        if (
            event != cv2.EVENT_LBUTTONDOWN
            or len(points) == 4
            or not (0 <= x < display_width and HEADER_HEIGHT <= y < HEADER_HEIGHT + display_height)
        ):
            return
        points.append((x * width / display_width, (y - HEADER_HEIGHT) * height / display_height))
        message = "Press ENTER to recognize, or R to start again." if len(points) == 4 else "Continue clockwise around the face."
        redraw()

    def choose_automatically():
        nonlocal mode, message
        points.clear()
        candidates = find_face_candidates(image)
        if candidates:
            points.extend(candidates[0].tolist())
            mode = "Automatic"
            message = "Check the outline, then press ENTER."
        else:
            mode = "Manual"
            message = "No reliable face found. Select corners manually."

    if not manual:
        choose_automatically()
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, on_mouse)
    redraw()
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord("q"), ord("Q")) or not window_open(WINDOW):
            return None
        if key in (ord("r"), ord("R"), ord("m"), ord("M")):
            points.clear()
            mode = "Manual"
            message = "Select the outer corners of one complete 3x3 face."
            redraw()
        elif key in (ord("a"), ord("A")):
            choose_automatically()
            redraw()
        elif key in (10, 13):
            try:
                return validate_corners(image, points), mode
            except ValueError as error:
                message = str(error)
                redraw()
