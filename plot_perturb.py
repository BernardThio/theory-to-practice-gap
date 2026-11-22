import csv
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt

# Read the perturbation results
with open("results/perturb.csv") as file:
    rows = list(csv.DictReader(file))

deltas = [0, 1e-6, 1e-4, 1e-2, 1e-1]
labels = ["0", "$10^{-6}$", "$10^{-4}$", "$10^{-2}$", "$10^{-1}$"]


# Median over the 3 seeds of one column, for a given m, lr and delta
def med(m: int, lr: float, delta: float, column: str) -> float:
    values = [
        float(row[column])
        for row in rows
        if int(row["m"]) == m and float(row["lr"]) == lr and float(row["delta"]) == delta
    ]
    return median(values)


# One panel per depth: error before and after training, against the noise size
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
for ax, m in zip(axes, [4, 8]):
    x = range(len(deltas))
    ax.semilogy(x, [med(m, 1e-3, d, "before") for d in deltas], "ko-", label="before training")
    ax.semilogy(x, [med(m, 1e-3, d, "after") for d in deltas], "o-", label="after, lr $10^{-3}$")
    ax.semilogy(x, [med(m, 1e-5, d, "after") for d in deltas], "s-", label="after, lr $10^{-5}$")
    ax.axhline(2.0**(-2*m - 2), color="gray", linestyle="--", label="Yarotsky")
    ax.set_xticks(x, labels)
    ax.set_xlabel(r"noise size $\delta$")
    ax.set_title(f"$m = {m}$")
axes[0].set_ylabel(r"$\|\mathrm{net} - x^2\|_\infty$ (median of 3 seeds)")
axes[0].legend()
fig.tight_layout()

Path("figures").mkdir(exist_ok=True)
fig.savefig("figures/perturb.png", dpi=200)
plt.show()