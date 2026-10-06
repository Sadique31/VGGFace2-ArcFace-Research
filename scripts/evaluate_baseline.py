import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# =========================
# Paths
# =========================
PROJECT = Path("/mnt/c/Users/ansar/Desktop/VGGFace2_ArcFace_Project")

INPUT_FILE = PROJECT / "results/baseline_similarity_scores.csv"
OUTPUT_FILE = PROJECT / "results/baseline_metrics.txt"

# =========================
# Load scores
# =========================
df = pd.read_csv(INPUT_FILE)

y_true = df["label"].to_numpy()
scores = df["cosine_similarity"].to_numpy()

# =========================
# ROC / AUC
# =========================
fpr, tpr, thresholds = roc_curve(y_true, scores)
auc = roc_auc_score(y_true, scores)

# =========================
# EER
# =========================
fnr = 1 - tpr

eer_index = np.nanargmin(
    np.abs(fpr - fnr)
)

eer = (fpr[eer_index] + fnr[eer_index]) / 2
eer_threshold = thresholds[eer_index]

# =========================
# Best threshold using F1
# =========================
best_f1 = -1
best_threshold = None
best_precision = None
best_recall = None

for threshold in thresholds:

    predictions = (
        scores >= threshold
    ).astype(int)

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    if f1 > best_f1:

        best_f1 = f1
        best_threshold = threshold
        best_precision = precision
        best_recall = recall

# =========================
# Confusion matrix
# =========================
predictions = (
    scores >= best_threshold
).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_true,
    predictions
).ravel()

far = fp / (fp + tn)
tar = tp / (tp + fn)

# =========================
# Print results
# =========================
print("\n========================================")
print("DEEPFACE ARCFACE BASELINE EVALUATION")
print("========================================")

print(f"\nEvaluation pairs: {len(df)}")
print(f"Genuine pairs: {(y_true == 1).sum()}")
print(f"Impostor pairs: {(y_true == 0).sum()}")

print("\n--- ROC ---")
print(f"ROC-AUC: {auc:.4f}")

print("\n--- EER ---")
print(f"EER: {eer:.4f}")
print(f"EER percentage: {eer * 100:.2f}%")
print(f"EER threshold: {eer_threshold:.4f}")

print("\n--- Best F1 Threshold ---")
print(f"Threshold: {best_threshold:.4f}")
print(f"Precision: {best_precision:.4f}")
print(f"Recall: {best_recall:.4f}")
print(f"F1-score: {best_f1:.4f}")

print("\n--- Threshold Performance ---")
print(f"TAR / TPR: {tar:.4f}")
print(f"FAR: {far:.4f}")

print("\n--- Confusion Matrix ---")
print(f"True Negatives : {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Positives : {tp}")

# =========================
# Save results
# =========================
with open(OUTPUT_FILE, "w") as f:

    f.write("DEEPFACE ARCFACE BASELINE EVALUATION\n")
    f.write("=" * 50 + "\n")
    f.write(f"Evaluation pairs: {len(df)}\n")
    f.write(f"ROC-AUC: {auc:.6f}\n")
    f.write(f"EER: {eer:.6f}\n")
    f.write(f"EER percentage: {eer * 100:.2f}%\n")
    f.write(f"EER threshold: {eer_threshold:.6f}\n")
    f.write(f"Best F1 threshold: {best_threshold:.6f}\n")
    f.write(f"Precision: {best_precision:.6f}\n")
    f.write(f"Recall: {best_recall:.6f}\n")
    f.write(f"F1-score: {best_f1:.6f}\n")
    f.write(f"TAR / TPR: {tar:.6f}\n")
    f.write(f"FAR: {far:.6f}\n")
    f.write(f"True Negatives: {tn}\n")
    f.write(f"False Positives: {fp}\n")
    f.write(f"False Negatives: {fn}\n")
    f.write(f"True Positives: {tp}\n")

print(f"\nResults saved: {OUTPUT_FILE}")
