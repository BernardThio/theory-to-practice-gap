import torch
import matplotlib.pyplot as plt
from torch import Tensor, relu
from sawtooth import g_s
from pathlib import Path

torch.set_default_dtype(torch.float64)

def f(x: Tensor, m: int) -> Tensor:
    return x - sum(g_s(x, s) / 4**s for s in range(1, m + 1))


xf = torch.linspace(0, 1, 2**16 + 1)
errors = []
for m in range(13):
    err = (f(xf, m) - xf**2).abs().max().item()
    print(m, err, err / 2.0**(-2*m - 2))
    errors.append(err)

x = torch.linspace(0, 1, 2**10 + 1)

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(x.numpy(), (x**2).numpy(), "k", linewidth=2, label="$x^2$")
for m in range(4):
    ax.plot(x.numpy(), f(x, m).numpy(), label=f"$f_{m}$")
ax.set_xlabel("$x$")
ax.legend()
fig.tight_layout()

Path("figures").mkdir(exist_ok=True)
fig.savefig("figures/interpolants.png", dpi=200)
plt.show()

fig, ax = plt.subplots(figsize=(6, 5))
ax.semilogy(range(13), errors, "o", label="measured")
ax.semilogy(range(13), [2.0**(-2*m - 2) for m in range(13)], "--", label="$2^{-2m-2}$")
ax.set_xlabel("$m$ (depth)")
ax.set_ylabel(r"$\|f_m - x^2\|_\infty$")
ax.legend()
fig.tight_layout()
fig.savefig("figures/error_formula.png", dpi=200)
plt.show()