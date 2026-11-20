import csv
from pathlib import Path

import matplotlib.pyplot as plt

# Read the training results
with open("results/trained.csv") as file:
    rows = list(csv.DictReader(file))

ms = [int(row["m"]) for row in rows]
best = [float(row["best"]) for row in rows]
med = [float(row["median"]) for row in rows]
yarotsky = [2.0**(-2*m - 2) for m in ms]

# Theory against practice, log scale
fig, ax = plt.subplots(figsize=(6, 5))
ax.semilogy(ms, yarotsky, "k--", label="Yarotsky (hand-built)")
ax.semilogy(ms, best, "o-", label="trained, best of 5")
ax.semilogy(ms, med, "s-", label="trained, median of 5")
ax.set_xticks(ms)
ax.set_xlabel("$m$ (depth)")
ax.set_ylabel(r"$\|\mathrm{net} - x^2\|_\infty$")
ax.legend()
fig.tight_layout()

Path("figures").mkdir(exist_ok=True)
fig.savefig("figures/gap.png", dpi=200)
plt.show()