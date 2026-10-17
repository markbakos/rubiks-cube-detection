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

## Technical overview

The project uses **Python**, **OpenCV** for image processing and native windows,
and **NumPy** for pixel arrays and calculations. The setup commands above create
an isolated `.venv` environment and install the libraries from `requirements.txt`.
Recognition uses geometric rules and color thresholds, without model training.

The image-processing pipeline works in these steps:

1. **Represent the image as pixels.** `image_processing.py` loads a three-channel
   BGR array: blue, green, and red values from 0 to 255 at each position.
   Transparent PNG pixels are blended onto white so transparency does not become
   a false black region. The original array is preserved.

2. **Add controlled noise.** Gaussian noise adds normally distributed random
   changes to each color channel; its standard deviation controls their strength.
   Salt-and-pepper noise replaces a chosen fraction of pixels with black or white.
   Values are clipped to 0–255. Every experiment starts from the original, with
   a fixed seed to reproduce the same random pattern.

3. **Filter and prepare.** A 5×5 Gaussian filter replaces each pixel with a
   weighted neighborhood average, giving nearby pixels more influence. It reduces
   random variation but also softens edges. A 3×3 median filter takes the middle
   neighborhood value, making isolated black or white outliers less influential.
   Detection always uses a blurred grayscale copy; color recognition keeps BGR.

4. **Find edges with Canny.** Edges are locations where image intensity changes
   sharply, such as sticker boundaries. Canny measures these changes, thins the
   response, and uses two thresholds: strong edges are retained, while weaker
   ones survive when connected to strong edges. This reduces disconnected noise.

5. **Find and validate a face.** In `face_detection.py`, the Hough transform
   finds straight segments supported by aligned edge pixels. Collinear segments
   are merged, and approximately parallel line pairs form quadrilaterals.
   Angular tolerance allows perspective convergence. Each candidate is temporarily
   straightened to 150×150: edges should follow the one-third and two-thirds grid
   divisions, with relatively quiet cell interiors. Grid support determines the
   ranking, with a small preference for larger faces; overlapping duplicates
   are removed. This helps reject individual stickers and background shapes.

6. **Correct perspective.** `manual_selection.py` lets the user accept the outline
   or supply four corners. Both modes use the same perspective transform, also
   called a homography: a 3×3 matrix mapping the photographed face onto a flat
   300×300 square. OpenCV interpolates source pixels to build the new image.
   Corner order determines its orientation.

7. **Sample sticker centers.** The corrected face has nine 100×100 cells.
   `color_detection.py` samples a central 40×40 region in each, avoiding plastic
   borders and neighboring stickers. It takes the median of each BGR channel,
   reducing the influence of isolated reflections and noisy pixels.

8. **Classify in HSV.** The representative colors are converted to hue (color
   family), saturation (color intensity), and value (brightness). This makes
   thresholds easier to interpret than mixtures of BGR values. White usually
   has low saturation; colored stickers are separated by hue. Red spans both ends
   of OpenCV's circular 0–179 hue range. Very dark or unsupported samples become
   `?`. HSV helps organize the rules, but does not remove lighting effects.

9. **Display and measure.** `main.py` displays the corrected face and matrix.
   `evaluation.py` first measures colors at known corners, isolating recognition
   from detection errors. It then evaluates the automatically selected face.
   Matching uses intersection-over-union: shared polygon area divided by combined
   area. A match requires at least 80% overlap. Reference orientation is aligned
   before comparing colors. Reports include correct cells, complete faces, and
   unknowns, which count as errors.

Fixed thresholds and visible grid boundaries remain practical limits. Noise
experiments measure controlled disturbances; additional real photographs are
still needed to assess different lighting, backgrounds, and viewing angles.

