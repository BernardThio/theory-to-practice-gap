import torch
from torch import nn
import csv
from pathlib import Path
from network import make_net, set_yarotsky_weights
from train import train, xf

torch.set_default_dtype(torch.float64)


# Add random noise of size delta to every weight and bias
def perturb(net: nn.Module, delta: float) -> None:
    with torch.no_grad():
        for p in net.parameters():
            p.add_(delta * torch.randn_like(p))


# Start at Yarotsky's weights plus noise, train with two learning rates, compare
if __name__ == "__main__":
    results = []
    for m in [4, 8]:
        for lr in [1e-3, 1e-5]:
            for delta in [0, 1e-6, 1e-4, 1e-2, 1e-1]:
                for seed in range(3):
                    torch.manual_seed(seed)
                    net = make_net(m)
                    set_yarotsky_weights(net, m)
                    perturb(net, delta)
                    with torch.no_grad():
                        before = (net(xf) - xf**2).abs().max().item()
                    after = train(net, steps=20000, lr=lr)
                    results.append((m, lr, delta, seed, before, after))
                    print(m, lr, delta, seed, before, after)

    Path("results").mkdir(exist_ok=True)
    with open("results/perturb.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["m", "lr", "delta", "seed", "before", "after"])
        writer.writerows(results)