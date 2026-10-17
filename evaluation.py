import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from color_detection import COLORS, recognize_face
from face_detection import find_face_candidates
from image_processing import load_image, process_image, rectify_face, validate_corners


def load_annotations(path):
    path = Path(path)
    records = json.loads(path.read_text())
    if not isinstance(records, list) or not records:
        raise ValueError("Annotations must contain a nonempty list of images.")
    loaded = []
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("image"), str):
            raise ValueError("Each annotation needs an image filename.")
        image = load_image(path.parent / record["image"])
        if not isinstance(record.get("faces"), list) or not record["faces"]:
            raise ValueError(f"No annotated faces for {record['image']}.")
        faces = []
        for face in record["faces"]:
            if not isinstance(face, dict):
                raise ValueError("Each face needs corners and a matrix.")
            corners = validate_corners(image, face.get("corners"))
            matrix = np.asarray(face.get("matrix"))
            if matrix.shape != (3, 3) or not np.isin(matrix, list(set(COLORS) - {"?"})).all():
                raise ValueError("Reference matrices must contain 3x3 known color codes.")
            faces.append((corners, matrix))
        loaded.append((record["image"], image, faces))
    return loaded


def evaluate_image(image, faces, noise="none", level=0, filter_method="none", seed=0):
    _, processed = process_image(image, noise, level, filter_method, seed)
    reference_correct = reference_unknown = reference_faces_correct = 0
    for corners, expected in faces:
        predicted = np.asarray(recognize_face(rectify_face(processed, corners)))
        reference_correct += int(np.sum(predicted == expected))
        reference_unknown += int(np.sum(predicted == "?"))
        reference_faces_correct += int(np.array_equal(predicted, expected))
    started = perf_counter()
    candidates = find_face_candidates(processed)
    elapsed = perf_counter() - started
    result = {
        "noise": noise, "level": level, "filter": filter_method, "seed": seed,
        "reference_correct": reference_correct, "reference_cells": len(faces) * 9,
        "reference_unknown": reference_unknown, "reference_faces_correct": reference_faces_correct,
        "reference_faces": len(faces), "detection": "none", "matched_face": "",
        "overlap": "", "automatic_correct": "", "automatic_unknown": "", "automatic_face_correct": "",
        "detection_seconds": round(elapsed, 3),
    }
    if candidates:
        candidate = candidates[0]
        overlaps = []
        for corners, expected in faces:
            intersection = cv2.intersectConvexConvex(candidate, corners)[0]
            overlaps.append(intersection / (cv2.contourArea(candidate) + cv2.contourArea(corners) - intersection))
        index = int(np.argmax(overlaps))
        result["overlap"] = round(overlaps[index], 4)
        result["detection"] = "unmatched"
        if overlaps[index] >= 0.8:
            corners, expected = faces[index]
            aligned = min((np.roll(candidate, shift, axis=0) for shift in range(4)), key=lambda quad: np.mean((quad - corners) ** 2))
            predicted = np.asarray(recognize_face(rectify_face(processed, aligned)))
            result.update(
                detection="matched", matched_face=index + 1,
                automatic_correct=int(np.sum(predicted == expected)),
                automatic_unknown=int(np.sum(predicted == "?")),
                automatic_face_correct=int(np.array_equal(predicted, expected)),
            )
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate cube recognition under reproducible noise and filtering.")
    parser.add_argument("annotations", nargs="?", type=Path, default=Path(__file__).parent / "images/annotations.json")
    parser.add_argument("--output", type=Path, default=Path("results/evaluation.csv"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    try:
        records = load_annotations(args.annotations)
        np.random.default_rng(args.seed)
        conditions = [("none", 0, "none")]
        for noise, levels, method in (
            ("gaussian", (5, 15, 30), "gaussian"),
            ("salt-pepper", (0.005, 0.02, 0.05), "median"),
        ):
            for level in levels:
                conditions.extend(((noise, level, "none"), (noise, level, method)))
        rows = []
        for filename, image, faces in records:
            for noise, level, method in conditions:
                row = {"image": filename, **evaluate_image(image, faces, noise, level, method, args.seed)}
                rows.append(row)
                print(f"{filename} | {noise} {level:g} | {method} | detection {row['detection']} | fixed corners {row['reference_correct']}/{row['reference_cells']}", flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved {len(rows)} experiments to {args.output}")
    except (OSError, ValueError, cv2.error) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
