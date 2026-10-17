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

Press M or R to select manually, or start with:

```bash
python main.py images/Rubik01.jpg --manual
```

For manual selection, click the four outer corners clockwise: top-left,
top-right, bottom-right, bottom-left. Press Enter to recognize.
If automatic detection finds no reliable face, manual selection opens instead.
Q, Escape, or closing either window exits.

The result shows the corrected face, sample regions, and a 3×3 color matrix.
Codes: F white, Z green, P red, K blue, N orange, S yellow, ? unknown.

Print CLI help or run the tests:

```bash
python main.py --help
python -m unittest discover -s tests -v
```

