import os
import cv2
import numpy as np
import pandas as pd

PROJECT = "/mnt/c/Users/ansar/Desktop/VGGFace2_ArcFace_Project"

PAIR_FILES = {
    "hard_genuine": os.path.join(
        PROJECT,
        "results/hard_pair_analysis/hard_genuine_pairs_top100.csv"
    ),
    "hard_impostor": os.path.join(
        PROJECT,
        "results/hard_pair_analysis/hard_impostor_pairs_top100.csv"
    )
}

OUTPUT_DIR = os.path.join(
    PROJECT,
    "results/hard_pair_analysis"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "hard_pair_quality_analysis.csv"
)


def calculate_image_quality(image_path):
    image = cv2.imread(image_path)

    if image is None:
        return {
            "width": np.nan,
            "height": np.nan,
            "face_area": np.nan,
            "blur": np.nan,
            "brightness": np.nan,
            "contrast": np.nan
        }

    height, width = image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = gray.mean()
    contrast = gray.std()

    # Since the images in data/aligned are already aligned 112x112 faces,
    # the full image area represents the available face crop area.
    face_area = width * height

    return {
        "width": width,
        "height": height,
        "face_area": face_area,
        "blur": blur,
        "brightness": brightness,
        "contrast": contrast
    }


all_results = []

for pair_type, csv_path in PAIR_FILES.items():

    print(f"\nProcessing: {pair_type}")

    df = pd.read_csv(csv_path)

    for index, row in df.iterrows():

        image1_path = row["image1"]
        image2_path = row["image2"]

        quality1 = calculate_image_quality(image1_path)
        quality2 = calculate_image_quality(image2_path)

        result = {
            "pair_type": pair_type,
            "image1": image1_path,
            "image2": image2_path,
            "cosine_similarity": row["cosine_similarity"],

            "image1_width": quality1["width"],
            "image1_height": quality1["height"],
            "image1_face_area": quality1["face_area"],
            "image1_blur": quality1["blur"],
            "image1_brightness": quality1["brightness"],
            "image1_contrast": quality1["contrast"],

            "image2_width": quality2["width"],
            "image2_height": quality2["height"],
            "image2_face_area": quality2["face_area"],
            "image2_blur": quality2["blur"],
            "image2_brightness": quality2["brightness"],
            "image2_contrast": quality2["contrast"],

            "mean_blur": np.mean([
                quality1["blur"],
                quality2["blur"]
            ]),

            "mean_brightness": np.mean([
                quality1["brightness"],
                quality2["brightness"]
            ]),

            "mean_contrast": np.mean([
                quality1["contrast"],
                quality2["contrast"]
            ]),

            "blur_difference": abs(
                quality1["blur"] - quality2["blur"]
            ),

            "brightness_difference": abs(
                quality1["brightness"] - quality2["brightness"]
            ),

            "contrast_difference": abs(
                quality1["contrast"] - quality2["contrast"]
            )
        }

        all_results.append(result)

        if (index + 1) % 25 == 0:
            print(f"  Processed {index + 1}/{len(df)} pairs")


results = pd.DataFrame(all_results)

results.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n========================================")
print("Hard-pair quality analysis completed")
print("========================================")

print(f"Total pairs analyzed: {len(results)}")
print(f"Genuine pairs: {(results['pair_type'] == 'hard_genuine').sum()}")
print(f"Impostor pairs: {(results['pair_type'] == 'hard_impostor').sum()}")

print(f"\nOutput:")
print(OUTPUT_FILE)

print("\nSummary:")
print(
    results.groupby("pair_type")[
        [
            "cosine_similarity",
            "mean_blur",
            "mean_brightness",
            "mean_contrast",
            "blur_difference",
            "brightness_difference",
            "contrast_difference"
        ]
    ].mean()
)
