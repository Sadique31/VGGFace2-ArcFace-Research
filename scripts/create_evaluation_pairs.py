import random
import pandas as pd
from pathlib import Path

# =========================
# Paths
# =========================
PROJECT = Path("/mnt/c/Users/ansar/Desktop/VGGFace2_ArcFace_Project")

INPUT_FILE = PROJECT / "results/arcface_embeddings/image_paths.csv"
OUTPUT_FILE = PROJECT / "results/evaluation_pairs.csv"

# =========================
# Configuration
# =========================
NUM_GENUINE = 10_000
NUM_IMPOSTOR = 10_000
SEED = 42

random.seed(SEED)

# =========================
# Load image paths
# =========================
df = pd.read_csv(INPUT_FILE)

# Extract identity from:
# .../train/n000002/0010_01.jpg
df["identity"] = df["image_path"].str.extract(
    r"/train/([^/]+)/"
)[0]

print("Images:", len(df))
print("Identities:", df["identity"].nunique())

# Group images by identity
groups = {
    identity: group["image_path"].tolist()
    for identity, group in df.groupby("identity")
}

identities = list(groups.keys())

# =========================
# Generate genuine pairs
# =========================
genuine_pairs = set()

while len(genuine_pairs) < NUM_GENUINE:

    identity = random.choice(identities)

    img1, img2 = random.sample(groups[identity], 2)

    pair = tuple(sorted([img1, img2]))

    genuine_pairs.add(pair)

print("Genuine pairs:", len(genuine_pairs))

# =========================
# Generate impostor pairs
# =========================
impostor_pairs = set()

while len(impostor_pairs) < NUM_IMPOSTOR:

    id1, id2 = random.sample(identities, 2)

    img1 = random.choice(groups[id1])
    img2 = random.choice(groups[id2])

    pair = (img1, img2)

    impostor_pairs.add(pair)

print("Impostor pairs:", len(impostor_pairs))

# =========================
# Create evaluation table
# =========================
rows = []

for img1, img2 in genuine_pairs:

    rows.append({
        "image1": img1,
        "image2": img2,
        "label": 1,
        "pair_type": "genuine"
    })

for img1, img2 in impostor_pairs:

    rows.append({
        "image1": img1,
        "image2": img2,
        "label": 0,
        "pair_type": "impostor"
    })

pairs = pd.DataFrame(rows)

# Shuffle pairs
pairs = pairs.sample(
    frac=1,
    random_state=SEED
).reset_index(drop=True)

# =========================
# Save
# =========================
pairs.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n==============================")
print("PAIR GENERATION COMPLETE")
print("==============================")
print("Total pairs:", len(pairs))
print("Genuine:", (pairs["label"] == 1).sum())
print("Impostor:", (pairs["label"] == 0).sum())
print("Saved:", OUTPUT_FILE)
