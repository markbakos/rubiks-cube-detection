# Rubik's Cube Face Recognition

A Python and OpenCV digital image processing project for recognizing the nine
colors of one visible 3×3 cube face.

## Current stage: initialization

The entry point checks that OpenCV and NumPy can be imported and prints their
versions. Recognition is not implemented yet.

Use Python 3.11 or newer and a desktop environment for the future OpenCV windows.
Install `opencv-python`, which includes GUI support, rather than its headless variant.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py
```

Print CLI help or run the entry point smoke test:

```bash
python main.py --help
python -m unittest discover -s tests -v
```

