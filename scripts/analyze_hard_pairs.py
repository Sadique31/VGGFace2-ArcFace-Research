import pandas as pd
from pathlib import Path

PROJECT = Path("/mnt/c/Users/ansar/Desktop/VGGFace2_ArcFace_Project")

INPUT_FILE = PROJECT / "results/baseline_similarity_scores.csv"
OUTPUT_DIR = PROJECT / "results/hard_pair_analysis"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# Load similarity results
# ---------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

genuine = df[df["label"] == 1].copy()
impostor = df[df["label"] == 0].copy()

# ---------------------------------------------------------
# Hard genuine pairs
# Same identity + lowest similarity
# ---------------------------------------------------------

hard_genuine = (
    genuine
    .sort_values("cosine_similarity", ascending=True)
    .head(100)
    .reset_index(drop=True)
)

# ---------------------------------------------------------
# Hard impostor pairs
# Different identities + highest similarity
# ---------------------------------------------------------

hard_impostor = (
    impostor
    .sort_values("cosine_similarity", ascending=False)
    .head(100)
    .reset_index(drop=True)
)

# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------

hard_genuine_file = OUTPUT_DIR / "hard_genuine_pairs_top100.csv"
hard_impostor_file = OUTPUT_DIR / "hard_impostor_pairs_top100.csv"

hard_genuine.to_csv(
    hard_genuine_file,
    index=False
)

hard_impostor.to_csv(
    hard_impostor_file,
    index=False
)

# ---------------------------------------------------------
# Print summary
# ---------------------------------------------------------

print("========================================")
print("HARD-PAIR ANALYSIS")
print("========================================")

print("\n--- Hard Genuine Pairs ---")
print("Same identity with lowest similarity")

print(
    hard_genuine[
        ["image1", "image2", "cosine_similarity"]
    ].head(20).to_string(index=False)
)

print("\n--- Hard Impostor Pairs ---")
print("Different identities with highest similarity")

print(
    hard_impostor[
        ["image1", "image2", "cosine_similarity"]
    ].head(20).to_string(index=False)
)

print("\n========================================")
print("SUMMARY")
print("========================================")

print(
    f"\nHard genuine similarity range: "
    f"{hard_genuine['cosine_similarity'].min():.4f} "
    f"to "
    f"{hard_genuine['cosine_similarity'].max():.4f}"
)

print(
    f"Hard impostor similarity range: "
    f"{hard_impostor['cosine_similarity'].min():.4f} "
    f"to "
    f"{hard_impostor['cosine_similarity'].max():.4f}"
)

print("\nSaved:")
print(hard_genuine_file)
print(hard_impostor_file)

print("\n========================================")
print("HARD-PAIR ANALYSIS COMPLETE")
print("========================================")
