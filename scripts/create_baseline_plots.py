import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
)

PROJECT = Path("/mnt/c/Users/ansar/Desktop/VGGFace2_ArcFace_Project")
INPUT_FILE = PROJECT / "results/baseline_similarity_scores.csv"
OUTPUT_DIR = PROJECT / "results/baseline_plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_csv(INPUT_FILE)

y_true = df["label"].to_numpy()
scores = df["cosine_similarity"].to_numpy()

genuine = scores[y_true == 1]
impostor = scores[y_true == 0]

print("========================================")
print("BASELINE VISUALIZATION SUITE")
print("========================================")
print(f"Total pairs   : {len(scores)}")
print(f"Genuine pairs : {len(genuine)}")
print(f"Impostor pairs: {len(impostor)}")


# ========================================================
# 1. Similarity Distribution
# ========================================================

plt.figure(figsize=(10, 6))

plt.hist(genuine, bins=80, alpha=0.6, density=True, label="Genuine")
plt.hist(impostor, bins=80, alpha=0.6, density=True, label="Impostor")

plt.axvline(
    0.2004,
    linestyle="--",
    linewidth=2,
    label="Best F1 Threshold = 0.2004"
)

plt.xlabel("Cosine Similarity")
plt.ylabel("Density")
plt.title("Genuine vs Impostor Similarity Distribution")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_similarity_distribution.png", dpi=300)
plt.close()


# ========================================================
# 2. ROC Curve
# ========================================================

fpr, tpr, roc_thresholds = roc_curve(y_true, scores)
auc = roc_auc_score(y_true, scores)

plt.figure(figsize=(8, 7))

plt.plot(
    fpr,
    tpr,
    linewidth=2,
    label=f"ArcFace (AUC = {auc:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — DeepFace ArcFace Baseline")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_roc_curve.png", dpi=300)
plt.close()


# ========================================================
# 3. Precision-Recall Curve
# ========================================================

precision, recall, pr_thresholds = precision_recall_curve(
    y_true,
    scores
)

average_precision = average_precision_score(
    y_true,
    scores
)

plt.figure(figsize=(8, 7))

plt.plot(
    recall,
    precision,
    linewidth=2,
    label=f"Average Precision = {average_precision:.4f}"
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve — DeepFace ArcFace Baseline")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_precision_recall_curve.png", dpi=300)
plt.close()


# ========================================================
# 4. EER Curve
# ========================================================

fnr = 1 - tpr

eer_index = np.nanargmin(
    np.abs(fpr - fnr)
)

eer = (fpr[eer_index] + fnr[eer_index]) / 2
eer_threshold = roc_thresholds[eer_index]

plt.figure(figsize=(9, 6))

plt.plot(
    roc_thresholds,
    fpr,
    label="FAR / FPR",
    linewidth=2
)

plt.plot(
    roc_thresholds,
    fnr,
    label="FRR / FNR",
    linewidth=2
)

plt.scatter(
    [eer_threshold],
    [eer],
    s=80,
    zorder=5,
    label=f"EER = {eer * 100:.2f}%"
)

plt.xlabel("Similarity Threshold")
plt.ylabel("Error Rate")
plt.title("Equal Error Rate (EER) Analysis")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_eer_curve.png", dpi=300)
plt.close()


# ========================================================
# 5. FAR and TAR vs Threshold
# ========================================================

thresholds = np.linspace(
    scores.min(),
    scores.max(),
    300
)

far_values = []
tar_values = []

for threshold in thresholds:

    predictions = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    far = fp / (fp + tn) if (fp + tn) else 0
    tar = tp / (tp + fn) if (tp + fn) else 0

    far_values.append(far)
    tar_values.append(tar)

plt.figure(figsize=(9, 6))

plt.plot(
    thresholds,
    far_values,
    label="FAR",
    linewidth=2
)

plt.plot(
    thresholds,
    tar_values,
    label="TAR / TPR",
    linewidth=2
)

plt.axvline(
    0.2004,
    linestyle="--",
    linewidth=2,
    label="Best F1 Threshold"
)

plt.xlabel("Similarity Threshold")
plt.ylabel("Rate")
plt.title("FAR and TAR vs Similarity Threshold")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "05_far_tar_vs_threshold.png", dpi=300)
plt.close()


# ========================================================
# 6. Precision, Recall and F1 vs Threshold
# ========================================================

precision_values = []
recall_values = []
f1_values = []

for threshold in thresholds:

    predictions = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    p = tp / (tp + fp) if (tp + fp) else 0
    r = tp / (tp + fn) if (tp + fn) else 0

    f1 = (
        2 * p * r / (p + r)
        if (p + r)
        else 0
    )

    precision_values.append(p)
    recall_values.append(r)
    f1_values.append(f1)

plt.figure(figsize=(9, 6))

plt.plot(
    thresholds,
    precision_values,
    label="Precision",
    linewidth=2
)

plt.plot(
    thresholds,
    recall_values,
    label="Recall",
    linewidth=2
)

plt.plot(
    thresholds,
    f1_values,
    label="F1-score",
    linewidth=2
)

plt.axvline(
    0.2004,
    linestyle="--",
    linewidth=2,
    label="Best F1 Threshold"
)

plt.xlabel("Similarity Threshold")
plt.ylabel("Score")
plt.title("Precision, Recall and F1 vs Threshold")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "06_precision_recall_f1_vs_threshold.png",
    dpi=300
)
plt.close()


# ========================================================
# 7. Box Plot
# ========================================================

plt.figure(figsize=(8, 6))

plt.boxplot(
    [genuine, impostor],
    tick_labels=["Genuine", "Impostor"],
    showfliers=False
)

plt.ylabel("Cosine Similarity")
plt.title("Genuine vs Impostor Similarity — Box Plot")
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "07_boxplot.png", dpi=300)
plt.close()


# ========================================================
# 8. Violin Plot
# ========================================================

plt.figure(figsize=(8, 6))

plt.violinplot(
    [genuine, impostor],
    showmedians=True,
    showextrema=True
)

plt.xticks(
    [1, 2],
    ["Genuine", "Impostor"]
)

plt.ylabel("Cosine Similarity")
plt.title("Genuine vs Impostor Similarity — Violin Plot")
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "08_violinplot.png", dpi=300)
plt.close()


# ========================================================
# 9. Cumulative Distribution Function
# ========================================================

genuine_sorted = np.sort(genuine)
impostor_sorted = np.sort(impostor)

genuine_cdf = (
    np.arange(1, len(genuine_sorted) + 1)
    / len(genuine_sorted)
)

impostor_cdf = (
    np.arange(1, len(impostor_sorted) + 1)
    / len(impostor_sorted)
)

plt.figure(figsize=(9, 6))

plt.plot(
    genuine_sorted,
    genuine_cdf,
    label="Genuine",
    linewidth=2
)

plt.plot(
    impostor_sorted,
    impostor_cdf,
    label="Impostor",
    linewidth=2
)

plt.axvline(
    0.2004,
    linestyle="--",
    linewidth=2,
    label="Best F1 Threshold"
)

plt.xlabel("Cosine Similarity")
plt.ylabel("Cumulative Probability")
plt.title("Cumulative Distribution of Similarity Scores")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "09_cdf.png", dpi=300)
plt.close()


# ========================================================
# 10. Similarity Score Histogram
# ========================================================

plt.figure(figsize=(10, 6))

plt.hist(
    genuine,
    bins=80,
    alpha=0.6,
    label="Genuine"
)

plt.hist(
    impostor,
    bins=80,
    alpha=0.6,
    label="Impostor"
)

plt.xlabel("Cosine Similarity")
plt.ylabel("Number of Pairs")
plt.title("Similarity Score Histogram")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "10_similarity_histogram.png",
    dpi=300
)
plt.close()


# ========================================================
# Finished
# ========================================================

print("\n========================================")
print("ALL BASELINE PLOTS GENERATED")
print("========================================")

print(f"ROC-AUC       : {auc:.4f}")
print(f"EER           : {eer:.4f}")
print(f"EER percentage: {eer * 100:.2f}%")
print(f"EER threshold : {eer_threshold:.4f}")

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nGenerated files:")

for file in sorted(OUTPUT_DIR.glob("*.png")):
    print(f"  {file.name}")

print("\n========================================")
print("DONE")
print("========================================")
