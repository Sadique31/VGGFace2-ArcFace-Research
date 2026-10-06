import os
import numpy as np
import pandas as pd

PROJECT = "/mnt/c/Users/ansar/Desktop/VGGFace2_ArcFace_Project"

LANDMARK_FILE = os.path.join(
    PROJECT,
    "metadata/train_landmarks.csv"
)

PAIR_FILE = os.path.join(
    PROJECT,
    "results/hard_pair_analysis/quality_comparison.csv"
)

OUTPUT_FILE = os.path.join(
    PROJECT,
    "results/hard_pair_analysis/pose_analysis.csv"
)


def calculate_pose_measure(points):
    """
    Estimate a relative horizontal pose/asymmetry measure.

    Points:
    P1 = left eye
    P2 = right eye
    P3 = nose
    P4 = left mouth corner
    P5 = right mouth corner

    The value is normalized by eye distance.
    This is a relative pose indicator, not a calibrated yaw angle.
    """

    left_eye = points[0]
    right_eye = points[1]
    nose = points[2]

    eye_midpoint = (left_eye + right_eye) / 2

    eye_distance = np.linalg.norm(
        right_eye - left_eye
    )

    if eye_distance == 0:
        return np.nan

    return (
        (nose[0] - eye_midpoint[0])
        / eye_distance
    )


print("Loading VGGFace2 landmarks...")

landmarks = pd.read_csv(LANDMARK_FILE)

print(f"Total landmark rows: {len(landmarks)}")

required_columns = [
    "NAME_ID",
    "P1X", "P1Y",
    "P2X", "P2Y",
    "P3X", "P3Y",
    "P4X", "P4Y",
    "P5X", "P5Y"
]

missing = [
    column
    for column in required_columns
    if column not in landmarks.columns
]

if missing:
    raise ValueError(
        f"Missing landmark columns: {missing}"
    )


# Create a direct NAME_ID -> landmark lookup.
landmark_lookup = {}

for _, row in landmarks.iterrows():

    name_id = str(row["NAME_ID"])

    points = np.array([
        [row["P1X"], row["P1Y"]],
        [row["P2X"], row["P2Y"]],
        [row["P3X"], row["P3Y"]],
        [row["P4X"], row["P4Y"]],
        [row["P5X"], row["P5Y"]]
    ], dtype=float)

    landmark_lookup[name_id] = points


print(
    f"Usable landmark entries: {len(landmark_lookup)}"
)


print("\nLoading pair data...")

pairs = pd.read_csv(PAIR_FILE)

print(f"Total pair rows: {len(pairs)}")


def get_name_id(image_path):
    """
    Convert a full aligned-image path into the original
    VGGFace2 NAME_ID format.

    Example:

    .../train/n004936/0136_04.jpg

    becomes:

    n004936/0136_04
    """

    normalized = str(image_path).replace("\\", "/")

    marker = "/train/"

    if marker not in normalized:
        return None

    relative = normalized.split(marker, 1)[1]

    relative = os.path.splitext(relative)[0]

    return relative


def get_pose(image_path):

    name_id = get_name_id(image_path)

    if name_id is None:
        return np.nan

    points = landmark_lookup.get(name_id)

    if points is None:
        return np.nan

    return calculate_pose_measure(points)


results = []

for index, row in pairs.iterrows():

    pose1 = get_pose(row["image1"])
    pose2 = get_pose(row["image2"])

    if pd.isna(pose1) or pd.isna(pose2):

        pose_difference = np.nan
        mean_absolute_pose = np.nan

    else:

        pose_difference = abs(
            pose1 - pose2
        )

        mean_absolute_pose = (
            abs(pose1) + abs(pose2)
        ) / 2

    results.append({
        "pair_type": row["pair_type"],
        "image1": row["image1"],
        "image2": row["image2"],
        "cosine_similarity": row["cosine_similarity"],
        "image1_pose": pose1,
        "image2_pose": pose2,
        "pose_difference": pose_difference,
        "mean_absolute_pose": mean_absolute_pose
    })

    if (index + 1) % 500 == 0:
        print(
            f"  Processed {index + 1}/{len(pairs)} pairs"
        )


results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n========================================")
print("Pose analysis completed")
print("========================================")

print(f"Output: {OUTPUT_FILE}")

print("\nComplete summary:")

summary = results_df.groupby("pair_type")[
    [
        "cosine_similarity",
        "pose_difference",
        "mean_absolute_pose"
    ]
].agg(
    ["mean", "median"]
)

print(
    summary.round(4)
)

print("\nMissing pose measurements:")

print(
    results_df[
        ["image1_pose", "image2_pose"]
    ].isna().sum()
)
