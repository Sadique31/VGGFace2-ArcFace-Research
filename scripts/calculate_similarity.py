import numpy as np
import pandas as pd
from pathlib import Path

# =========================
# Paths
# =========================
PROJECT = Path("/mnt/c/Users/ansar/Desktop/VGGFace2_ArcFace_Project")

EMBEDDINGS_FILE = PROJECT / "results/arcface_embeddings/embeddings.npy"
PATHS_FILE = PROJECT / "results/arcface_embeddings/image_paths.csv"
PAIRS_FILE = PROJECT / "results/evaluation_pairs.csv"
OUTPUT_FILE = PROJECT / "results/baseline_similarity_scores.csv"

# =========================
# Load embeddings
# =========================
embeddings = np.load(EMBEDDINGS_FILE, mmap_mode="r")

paths_df = pd.read_csv(PATHS_FILE)
pairs_df = pd.read_csv(PAIRS_FILE)

# Map image path → embedding index
path_to_index = {
    path: index
    for index, path in enumerate(paths_df["image_path"])
}

print("Embeddings:", embeddings.shape)
print("Pairs:", len(pairs_df))

# =========================
# Calculate cosine similarity
# =========================
scores = []

for _, row in pairs_df.iterrows():

    idx1 = path_to_index[row["image1"]]
    idx2 = path_to_index[row["image2"]]

    emb1 = embeddings[idx1]
    emb2 = embeddings[idx2]

    # Normalize embeddings
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)

    similarity = np.dot(emb1, emb2) / (norm1 * norm2)

    scores.append(float(similarity))

# =========================
# Save results
# =========================
pairs_df["cosine_similarity"] = scores

pairs_df.to_csv(
    OUTPUT_FILE,
    index=False
)

# =========================
# Basic statistics
# =========================
genuine = pairs_df[
    pairs_df["label"] == 1
]["cosine_similarity"]

impostor = pairs_df[
    pairs_df["label"] == 0
]["cosine_similarity"]

print("\n==============================")
print("SIMILARITY CALCULATION COMPLETE")
print("==============================")

print("Total pairs:", len(pairs_df))

print("\nGenuine similarity:")
print("Mean:", round(genuine.mean(), 4))
print("Min :", round(genuine.min(), 4))
print("Max :", round(genuine.max(), 4))

print("\nImpostor similarity:")
print("Mean:", round(impostor.mean(), 4))
print("Min :", round(impostor.min(), 4))
print("Max :", round(impostor.max(), 4))

print("\nSaved:", OUTPUT_FILE)
