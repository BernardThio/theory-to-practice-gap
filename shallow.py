import torch
from torch import nn
from statistics import median
from train import train
import csv
from pathlib import Path

torch.set_default_dtype(torch.float64)


# One hidden layer of N neurons
def make_shallow(N: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(1, N), nn.ReLU(), nn.Linear(N, 1))


# Train shallow networks with the same weight count as depth m
if __name__ == "__main__":
    results = []
    for m in range(1, 13):
        N = round((20*m - 8) / 3)
        errs = []
        for seed in range(5):
            torch.manual_seed(seed)
            net = make_shallow(N)
            errs.append(train(net))
        best, med = min(errs), median(errs)
        results.append((m, N, best, med))
        print(m, N, best, med)
    Path("results").mkdir(exist_ok=True)
    with open("results/shallow.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["m", "N", "best", "median"])
        writer.writerows(results)