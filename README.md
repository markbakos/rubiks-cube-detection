# Rubik's Cube Face Recognition

A Python and OpenCV digital image processing project for recognizing the nine
colors of one visible 3×3 cube face.

Use Python 3.11 or newer and a graphical desktop for the OpenCV windows.
Install `opencv-python`, which includes GUI support, rather than its headless variant.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py path/to/cube.jpg
```

The application automatically outlines one detected face. Press Enter to accept
the outline and recognize its nine colors. A runs automatic selection again.

Press M to select manually, or start with:

```bash
python main.py images/Rubik01.jpg --manual
```

For manual selection, click the four outer corners clockwise: top-left,
top-right, bottom-right, bottom-left. Press Enter to recognize.
R resets the current mode: it clears manual points or reruns automatic detection.
If automatic detection finds no reliable face, manual selection opens instead.
Q, Escape, or closing the selection or result window exits.

The result shows the corrected face, sample regions, and a 3×3 color matrix.
Codes: F white, Z green, P red, K blue, N orange, S yellow, ? unknown.

G cycles Gaussian noise levels 5, 15, and 30. P cycles salt-and-pepper densities
0.5%, 2%, and 5%. F cycles no filter, Gaussian filtering, and median filtering.
O restores the original photograph. D toggles a comparison of the original,
noisy, filtered, and edge images. These controls also work in the result window.
Noise changes rerun automatic selection; manual results keep the selected corners.
After a fallback to manual selection, press A to retry with the chosen filter.

Start with a repeatable noise experiment:

```bash
python main.py images/Rubik03.jpg --noise salt-pepper --level 0.05 --filter median
python main.py images/Rubik03.jpg --noise gaussian --level 30 --filter gaussian --seed 0
```

Run the default 52-experiment batch evaluation without opening windows:

```bash
python evaluation.py
python evaluation.py path/to/annotations.json --output results/custom.csv --seed 0
```

The default report is `results/evaluation.csv`. It compares recognition at fixed
reference corners with automatic detection and recognition. `matched` means the
selected face overlaps an annotated face by at least 80%; `unmatched` leaves
automatic accuracy blank because the selected region has no matching reference.
`none` means detection failed. Unknown colors count as incorrect.

For additional photographs, use the structure in `images/annotations.json`.
Image filenames are relative to the annotation file. Each recorded face needs
four clockwise corners and its independently checked 3×3 color matrix.

Print CLI help or run the tests:

```bash
python main.py --help
python -m unittest discover -s tests -v
```

