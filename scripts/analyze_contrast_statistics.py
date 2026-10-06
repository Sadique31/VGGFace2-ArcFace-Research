import pandas as pd
from scipy.stats import mannwhitneyu

INPUT = "results/hard_pair_analysis/quality_comparison.csv"
OUTPUT = "results/hard_pair_analysis/contrast_statistics.txt"

df = pd.read_csv(INPUT)

def compare(group1, group2, label1, label2):
    x = df[df["pair_type"] == group1]["contrast_difference"].dropna()
    y = df[df["pair_type"] == group2]["contrast_difference"].dropna()

    u, p = mannwhitneyu(x, y, alternative="two-sided")

    nx, ny = len(x), len(y)

    pooled_std = (
        ((nx - 1) * x.std()**2 + (ny - 1) * y.std()**2)
        / (nx + ny - 2)
    ) ** 0.5

    d = (x.mean() - y.mean()) / pooled_std

    print(f"\n{label1} vs {label2}")
    print(f"{label1}: N={nx}, Mean={x.mean():.4f}, Median={x.median():.4f}")
    print(f"{label2}: N={ny}, Mean={y.mean():.4f}, Median={y.median():.4f}")
    print(f"Mann-Whitney U = {u:.2f}")
    print(f"p-value = {p:.6g}")
    print(f"Cohen's d = {d:.4f}")

    if p < 0.001:
        print("Interpretation: Highly statistically significant (p < 0.001)")
    elif p < 0.05:
        print("Interpretation: Statistically significant (p < 0.05)")
    else:
        print("Interpretation: Not statistically significant (p >= 0.05)")

    return {
        "u": u,
        "p": p,
        "d": d,
        "n1": nx,
        "n2": ny,
        "mean1": x.mean(),
        "mean2": y.mean(),
        "median1": x.median(),
        "median2": y.median(),
    }


print("Loading quality analysis...")
print(f"Total rows: {len(df)}")

results = []

results.append(
    compare(
        "hard_genuine",
        "random_genuine",
        "hard_genuine",
        "random_genuine"
    )
)

results.append(
    compare(
        "hard_impostor",
        "random_impostor",
        "hard_impostor",
        "random_impostor"
    )
)

with open(OUTPUT, "w") as f:
    f.write("Contrast Statistical Analysis\n")
    f.write("=" * 50 + "\n\n")

    comparisons = [
        ("Hard Genuine vs Random Genuine", results[0]),
        ("Hard Impostor vs Random Impostor", results[1]),
    ]

    for title, r in comparisons:
        f.write(title + "\n")
        f.write("-" * len(title) + "\n")
        f.write(f"N1 = {r['n1']}\n")
        f.write(f"N2 = {r['n2']}\n")
        f.write(f"Mean 1 = {r['mean1']:.4f}\n")
        f.write(f"Mean 2 = {r['mean2']:.4f}\n")
        f.write(f"Median 1 = {r['median1']:.4f}\n")
        f.write(f"Median 2 = {r['median2']:.4f}\n")
        f.write(f"Mann-Whitney U = {r['u']:.2f}\n")
        f.write(f"p-value = {r['p']:.6g}\n")
        f.write(f"Cohen's d = {r['d']:.4f}\n\n")

print("\n========================================")
print("Contrast statistical analysis completed")
print("========================================")
print(f"Saved: {OUTPUT}")
