import os
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu

PROJECT = "/mnt/c/Users/ansar/Desktop/VGGFace2_ArcFace_Project"

INPUT_FILE = os.path.join(
    PROJECT,
    "results/hard_pair_analysis/pose_analysis.csv"
)

OUTPUT_FILE = os.path.join(
    PROJECT,
    "results/hard_pair_analysis/pose_statistics.txt"
)


def cohens_d(group1, group2):
    group1 = np.asarray(group1, dtype=float)
    group2 = np.asarray(group2, dtype=float)

    n1 = len(group1)
    n2 = len(group2)

    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)

    pooled_std = np.sqrt(
        ((n1 - 1) * var1 + (n2 - 1) * var2)
        / (n1 + n2 - 2)
    )

    if pooled_std == 0:
        return np.nan

    return (np.mean(group1) - np.mean(group2)) / pooled_std


print("Loading pose analysis...")

df = pd.read_csv(INPUT_FILE)

print(f"Total pairs: {len(df)}")

results = []

comparisons = [
    (
        "Genuine: hard vs random",
        "hard_genuine",
        "random_genuine"
    ),
    (
        "Impostor: hard vs random",
        "hard_impostor",
        "random_impostor"
    )
]

for name, group_a_name, group_b_name in comparisons:

    group_a = df[
        df["pair_type"] == group_a_name
    ]["mean_absolute_pose"].dropna()

    group_b = df[
        df["pair_type"] == group_b_name
    ]["mean_absolute_pose"].dropna()

    statistic, p_value = mannwhitneyu(
        group_a,
        group_b,
        alternative="two-sided"
    )

    d = cohens_d(group_a, group_b)

    mean_a = group_a.mean()
    mean_b = group_b.mean()

    median_a = group_a.median()
    median_b = group_b.median()

    print("\n========================================")
    print(name)
    print("========================================")

    print(f"{group_a_name}:")
    print(f"  N = {len(group_a)}")
    print(f"  Mean = {mean_a:.4f}")
    print(f"  Median = {median_a:.4f}")

    print(f"\n{group_b_name}:")
    print(f"  N = {len(group_b)}")
    print(f"  Mean = {mean_b:.4f}")
    print(f"  Median = {median_b:.4f}")

    print("\nStatistical test:")
    print(f"  Mann-Whitney U = {statistic:.2f}")
    print(f"  p-value = {p_value:.6g}")
    print(f"  Cohen's d = {d:.4f}")

    if p_value < 0.001:
        significance = "Highly statistically significant (p < 0.001)"
    elif p_value < 0.01:
        significance = "Statistically significant (p < 0.01)"
    elif p_value < 0.05:
        significance = "Statistically significant (p < 0.05)"
    else:
        significance = "Not statistically significant (p >= 0.05)"

    print(f"  Interpretation: {significance}")

    results.append({
        "comparison": name,
        "group_a": group_a_name,
        "group_b": group_b_name,
        "n_a": len(group_a),
        "n_b": len(group_b),
        "mean_a": mean_a,
        "mean_b": mean_b,
        "median_a": median_a,
        "median_b": median_b,
        "mann_whitney_u": statistic,
        "p_value": p_value,
        "cohens_d": d,
        "interpretation": significance
    })


results_df = pd.DataFrame(results)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

    f.write("POSE STATISTICAL ANALYSIS\n")
    f.write("=" * 60 + "\n\n")

    for _, row in results_df.iterrows():

        f.write(f"Comparison: {row['comparison']}\n")
        f.write("-" * 60 + "\n")

        f.write(
            f"{row['group_a']}: "
            f"N={row['n_a']}, "
            f"Mean={row['mean_a']:.4f}, "
            f"Median={row['median_a']:.4f}\n"
        )

        f.write(
            f"{row['group_b']}: "
            f"N={row['n_b']}, "
            f"Mean={row['mean_b']:.4f}, "
            f"Median={row['median_b']:.4f}\n"
        )

        f.write(
            f"Mann-Whitney U: "
            f"{row['mann_whitney_u']:.2f}\n"
        )

        f.write(
            f"p-value: "
            f"{row['p_value']:.6g}\n"
        )

        f.write(
            f"Cohen's d: "
            f"{row['cohens_d']:.4f}\n"
        )

        f.write(
            f"Interpretation: "
            f"{row['interpretation']}\n\n"
        )

print("\n========================================")
print("Pose statistical analysis completed")
print("========================================")
print(f"Saved: {OUTPUT_FILE}")
