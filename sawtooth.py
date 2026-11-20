import torch
import matplotlib.pyplot as plt
from torch import Tensor, relu
from pathlib import Path

torch.set_default_dtype(torch.float64)


# Hat function: 0 at x=0, 1 at x=1/2, 0 at x=1
def g(x: Tensor) -> Tensor:
    return 2*relu(x) - 4*relu(x - 0.5) + 2*relu(x - 1)


# Apply g s times: a sawtooth with 2^(s-1) teeth
def g_s(x: Tensor, s: int) -> Tensor:
    for _ in range(s):
        x = g(x)
    return x


# Plot g_1 to g_4 when this file is run directly
if __name__ == "__main__":
    x = torch.linspace(0, 1, 2**10 + 1)  # grid hits every corner

    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(6, 8))
    for s, ax in enumerate(axes, start=1):
        ax.plot(x.numpy(), g_s(x, s).numpy())
        ax.set_title(f"$g_{s}$")
    axes[-1].set_xlabel("$x$")
    fig.tight_layout()

    # Save before showing
    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/sawtooth.png", dpi=200)
    plt.show()