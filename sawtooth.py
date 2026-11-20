import torch
import matplotlib.pyplot as plt
from torch import Tensor, relu
from pathlib import Path

torch.set_default_dtype(torch.float64)

def g(x: Tensor) -> Tensor:
    return 2*relu(x) - 4*relu(x - 0.5) + 2*relu(x - 1)

def g_s(x: Tensor, s: int) -> Tensor:
    for _ in range(s):
        x = g(x)
    return x

if __name__ == "__main__":
    x = torch.linspace(0, 1, 2**10 + 1)

    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(6, 8))
    for s, ax in enumerate(axes, start=1):
        ax.plot(x.numpy(), g_s(x, s).numpy())
        ax.set_title(f"$g_{s}$")
    axes[-1].set_xlabel("$x$")
    fig.tight_layout()

    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/sawtooth.png", dpi=200)
    plt.show()