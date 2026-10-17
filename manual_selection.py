import cv2
import numpy as np

from image_processing import process_image, validate_corners
from face_detection import find_face_candidates


WINDOW = "Cube face selection"
HEADER_HEIGHT = 150
COMPARISON_WINDOW = "Noise and filtering comparison"


def window_open(name):
    try:
        return cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        return False


def display_size(image):
    height, width = image.shape[:2]
    scale = min(1.0, 1000 / width, 550 / height)
    return max(1, round(width * scale)), max(1, round(height * scale))


def change_experiment(settings, key):
    key = chr(key).lower()
    if key in ("g", "p"):
        noise, levels = ("gaussian", (5, 15, 30)) if key == "g" else ("salt-pepper", (0.005, 0.02, 0.05))
        index = levels.index(settings["level"]) + 1 if settings["noise"] == noise and settings["level"] in levels else 0
        settings.update(noise=noise, level=levels[index % len(levels)])
    elif key == "f":
        methods = ("none", "gaussian", "median")
        settings["filter_method"] = methods[(methods.index(settings["filter_method"]) + 1) % len(methods)]
    elif key == "o":
        settings.update(noise="none", level=0, filter_method="none")
    else:
        return False
    return True


def draw_selection(image, points, mode, message, settings):
    height, width = image.shape[:2]
    display_width, display_height = display_size(image)
    display = cv2.resize(image, (display_width, display_height))
    canvas_width = max(660, display_width)
    canvas = np.full((display_height + HEADER_HEIGHT, canvas_width, 3), 28, dtype=np.uint8)
    canvas[HEADER_HEIGHT:, :display_width] = display
    level = f"{settings['level'] * 100:g}%" if settings["noise"] == "salt-pepper" else f"{settings['level']:g}"
    for y, text, color in (
        (25, "Click corners: 1 TL -> 2 TR -> 3 BR -> 4 BL" if mode == "Manual" else "Automatic face selected - ENTER accept or M select manually", (245, 245, 245)),
        (50, "ENTER recognize | A auto | M manual | R reset | Q / ESC quit", (190, 190, 190)),
        (75, "G Gaussian | P salt-pepper | F filter | O original | D compare", (190, 190, 190)),
        (100, f"Noise: {settings['noise']} {level} | Filter: {settings['filter_method']} | Seed: {settings['seed']}", (190, 190, 190)),
        (128, f"{mode} {len(points)}/4  {message}", (80, 220, 255)),
    ):
        cv2.putText(canvas, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    if len(points):
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
    return canvas


def draw_comparison(original, noisy, processed):
    gray = cv2.GaussianBlur(cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.cvtColor(cv2.Canny(gray, 40, 120), cv2.COLOR_GRAY2BGR)
    canvas = np.full((660, 880, 3), 28, dtype=np.uint8)
    for index, (label, image) in enumerate((
        ("Original", original), ("Noisy", noisy), ("Filtered", processed), ("Edges after filtering", edges),
    )):
        x, y = 10 + (index % 2) * 440, 30 + (index // 2) * 330
        cv2.putText(canvas, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
        height, width = image.shape[:2]
        scale = min(420 / width, 280 / height)
        preview = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))))
        canvas[y + 10:y + 10 + preview.shape[0], x:x + preview.shape[1]] = preview
    return canvas


def toggle_comparison(original, settings):
    if window_open(COMPARISON_WINDOW):
        cv2.destroyWindow(COMPARISON_WINDOW)
    else:
        noisy, processed = process_image(original, **settings)
        cv2.imshow(COMPARISON_WINDOW, draw_comparison(original, noisy, processed))


def select_face(image, manual=False, settings=None):
    if settings is None:
        settings = {"noise": "none", "level": 0, "filter_method": "none", "seed": 0}
    noisy, processed = process_image(image, **settings)
    height, width = image.shape[:2]
    display_width, display_height = display_size(image)
    points = []
    mode = "Manual"
    message = "Select the outer corners of one complete 3x3 face."

    def redraw():
        cv2.imshow(WINDOW, draw_selection(processed, points, mode, message, settings))

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
        candidates = find_face_candidates(processed)
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
    if window_open(COMPARISON_WINDOW):
        cv2.imshow(COMPARISON_WINDOW, draw_comparison(image, noisy, processed))
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord("q"), ord("Q")) or not window_open(WINDOW):
            return None
        if change_experiment(settings, key):
            noisy, processed = process_image(image, **settings)
            if mode == "Automatic":
                choose_automatically()
            redraw()
            if window_open(COMPARISON_WINDOW):
                cv2.imshow(COMPARISON_WINDOW, draw_comparison(image, noisy, processed))
            continue
        if key in (ord("d"), ord("D")):
            toggle_comparison(image, settings)
            continue
        if key in (ord("r"), ord("R")):
            key = ord("a") if mode == "Automatic" else ord("m")
        if key in (ord("m"), ord("M")):
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
