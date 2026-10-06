import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT = Path("/mnt/c/Users/ansar/Desktop/VGGFace2_ArcFace_Project")

INPUT_FILE = PROJECT / "results/baseline_similarity_scores.csv"
OUTPUT_FILE = PROJECT / "results/baseline_similarity_distribution.png"

df = pd.read_csv(INPUT_FILE)

genuine = df[df["label"] == 1]["cosine_similarity"]
impostor = df[df["label"] == 0]["cosine_similarity"]

plt.figure(figsize=(10, 6))

plt.hist(
    genuine,
    bins=50,
    alpha=0.6,
    label="Genuine (Same Identity)"
)

plt.hist(
    impostor,
    bins=50,
    alpha=0.6,
    label="Impostor (Different Identity)"
)

plt.xlabel("Cosine Similarity")
plt.ylabel("Number of Pairs")
plt.title("DeepFace ArcFace Baseline Similarity Distribution")
plt.legend()
plt.grid(alpha=0.2)

plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=300)
plt.close()

print("Plot saved:", OUTPUT_FILE)
