import os
import random
import cv2
import numpy as np
import pandas as pd

PROJECT = "/mnt/c/Users/ansar/Desktop/VGGFace2_ArcFace_Project"

EVALUATION_FILE = os.path.join(
    PROJECT,
    "results/baseline_similarity_scores.csv"
)

HARD_QUALITY_FILE = os.path.join(
    PROJECT,
    "results/hard_pair_analysis/hard_pair_quality_analysis.csv"
)

OUTPUT_FILE = os.path.join(
    PROJECT,
    "results/hard_pair_analysis/quality_comparison.csv"
)

RANDOM_SEED = 42
SAMPLE_SIZE = 1000


def calculate_image_quality(image_path):
    image = cv2.imread(image_path)

    if image is None:
        return {
            "blur": np.nan,
            "brightness": np.nan,
            "contrast": np.nan
        }

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return {
        "blur": cv2.Laplacian(gray, cv2.CV_64F).var(),
        "brightness": gray.mean(),
        "contrast": gray.std()
    }


def calculate_pair_quality(row):
    q1 = calculate_image_quality(row["image1"])
    q2 = calculate_image_quality(row["image2"])

    return {
        "mean_blur": np.mean([q1["blur"], q2["blur"]]),
        "mean_brightness": np.mean([q1["brightness"], q2["brightness"]]),
        "mean_contrast": np.mean([q1["contrast"], q2["contrast"]]),
        "blur_difference": abs(q1["blur"] - q2["blur"]),
        "brightness_difference": abs(q1["brightness"] - q2["brightness"]),
        "contrast_difference": abs(q1["contrast"] - q2["contrast"])
    }


print("Loading evaluation pairs...")

evaluation = pd.read_csv(EVALUATION_FILE)

random.seed(RANDOM_SEED)

genuine = evaluation[evaluation["label"] == 1]
impostor = evaluation[evaluation["label"] == 0]

random_genuine = genuine.sample(
    n=min(SAMPLE_SIZE, len(genuine)),
    random_state=RANDOM_SEED
).copy()

random_impostor = impostor.sample(
    n=min(SAMPLE_SIZE, len(impostor)),
    random_state=RANDOM_SEED
).copy()

print(f"Random genuine pairs: {len(random_genuine)}")
print(f"Random impostor pairs: {len(random_impostor)}")


results = []

for pair_type, df in [
    ("random_genuine", random_genuine),
    ("random_impostor", random_impostor)
]:

    print(f"\nProcessing {pair_type}...")

    for index, (_, row) in enumerate(df.iterrows(), start=1):

        quality = calculate_pair_quality(row)

        results.append({
            "pair_type": pair_type,
            "image1": row["image1"],
            "image2": row["image2"],
            "cosine_similarity": row["cosine_similarity"],
            **quality
        })

        if index % 100 == 0:
            print(f"  Processed {index}/{len(df)} pairs")


random_results = pd.DataFrame(results)

# Keep only the same quality columns used for the hard-pair analysis.
hard_results = pd.read_csv(HARD_QUALITY_FILE)

hard_results = hard_results[
    [
        "pair_type",
        "image1",
        "image2",
        "cosine_similarity",
        "mean_blur",
        "mean_brightness",
        "mean_contrast",
        "blur_difference",
        "brightness_difference",
        "contrast_difference"
    ]
].copy()

combined = pd.concat(
    [random_results, hard_results],
    ignore_index=True
)

combined.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n========================================")
print("Quality control-group analysis completed")
print("========================================")

print(f"Total pairs: {len(combined)}")

summary = combined.groupby("pair_type")[
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

print("\nComplete comparison:")
print(summary.round(3))

print("\nOutput:")
print(OUTPUT_FILE)
