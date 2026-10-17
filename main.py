import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Rubik's Cube single-face recognition (initial setup)."
    )
    parser.parse_args()

    import cv2
    import numpy as np

    print(f"OpenCV {cv2.__version__} | NumPy {np.__version__}")
    print("Setup ready. Manual single-face recognition is the next milestone.")


if __name__ == "__main__":
    main()
