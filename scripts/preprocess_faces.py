import csv
from pathlib import Path

import cv2
import numpy as np


# Paths
INPUT_DIR = Path("data/selected")
OUTPUT_DIR = Path("data/aligned")
LANDMARKS_FILE = Path("metadata/train_landmarks.csv")
BBOX_FILE = Path("metadata/train_bounding_boxes.csv")

# ArcFace 112x112 reference landmarks
DST_POINTS = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)


def load_landmarks():
    """Load VGGFace2 5-point landmarks into a dictionary."""
    landmarks = {}

    with LANDMARKS_FILE.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            name_id = row["NAME_ID"]

            points = np.array([
                [float(row["P1X"]), float(row["P1Y"])],
                [float(row["P2X"]), float(row["P2Y"])],
                [float(row["P3X"]), float(row["P3Y"])],
                [float(row["P4X"]), float(row["P4Y"])],
                [float(row["P5X"]), float(row["P5Y"])]
            ], dtype=np.float32)

            landmarks[name_id] = points

    return landmarks

def load_bounding_boxes():
    """Load VGGFace2 bounding boxes into a dictionary."""
    boxes = {}

    with BBOX_FILE.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            name_id = row["NAME_ID"]

            boxes[name_id] = (
                float(row["X"]),
                float(row["Y"]),
                float(row["W"]),
                float(row["H"])
            )

    return boxes

def align_face(image, src_points):
    """Align a face to the standard ArcFace 112x112 template."""
    transform, _ = cv2.estimateAffinePartial2D(
        src_points,
        DST_POINTS,
        method=cv2.LMEDS
    )

    if transform is None:
        return None

    aligned = cv2.warpAffine(
        image,
        transform,
        (112, 112),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT
    )

    return aligned


print("Loading landmarks...")
landmarks = load_landmarks()
boxes = load_bounding_boxes()
print(f"Landmark records loaded: {len(landmarks)}")

print(f"Bounding-box records loaded: {len(boxes)}")

import time

images = list(INPUT_DIR.rglob("*.jpg"))

print(f"Benchmark images: {len(images)}")

start = time.time()
processed = 0
skipped = 0

for image_path in images:
    relative = image_path.relative_to(INPUT_DIR)
    name_id = relative.with_suffix("").as_posix().replace("train/", "", 1)

    image = cv2.imread(str(image_path))

    if image is None:
        skipped += 1
        continue

    src_points = landmarks[name_id]

    x, y, w, h = map(int, boxes[name_id])

    # Original image dimensions
    img_h, img_w = image.shape[:2]

    # Clip bounding box to image boundaries
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(img_w, x + w)
    y2 = min(img_h, y + h)

    # Skip invalid bounding boxes
    if x2 <= x1 or y2 <= y1:
        skipped += 1
        continue

    # Crop face
    face_crop = image[y1:y2, x1:x2]

    # Shift landmarks relative to cropped face
    src_points_crop = src_points - np.array(
        [x1, y1],
        dtype=np.float32
    )

    aligned = align_face(face_crop, src_points_crop)

    if aligned is not None:
        output_path = OUTPUT_DIR / relative
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), aligned)
        processed += 1
    else:
        skipped += 1

elapsed = time.time() - start

print("\nBenchmark complete.")
print("Processed:", processed)
print("Skipped:", skipped)
print("Time:", round(elapsed, 2), "seconds")

if processed > 0:
    speed = processed / elapsed

    print("Images/second:", round(speed, 2))
    print(
        "Estimated time for 129,465 images:",
        round(129465 / speed / 3600, 2),
        "hours"
    )