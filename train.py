from statistics import median

import torch
from torch import nn
import csv
from pathlib import Path

from network import make_net

torch.set_default_dtype(torch.float64)

# Training data: 1000 random points in [0,1] and their squares
torch.manual_seed(0)
x_train = torch.rand(1000, 1)
y_train = x_train**2

# Fine grid for measuring the sup error
xf = torch.linspace(0, 1, 2**16 + 1).reshape(-1, 1)


# Train a depth-m network from random weights and return its sup error
def train(net: nn.Module, steps: int = 20000) -> float:
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    for _ in range(steps):
        optimizer.zero_grad()
        loss = loss_fn(net(x_train), y_train)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        return (net(xf) - xf**2).abs().max().item()


# Run every depth with 5 seeds; keep the best and median sup error
if __name__ == "__main__":
    results = []
    for m in range(1, 13):
        errs = []
        for seed in range(5):
            torch.manual_seed(seed)
            net = make_net(m)
            errs.append(train(net))
        best, med = min(errs), median(errs)
        results.append((m, best, med))
        print(m, best, med, 2.0**(-2*m - 2))
    
    Path("results").mkdir(exist_ok=True)
    with open("results/trained.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["m", "best", "median"])
        writer.writerows(results)