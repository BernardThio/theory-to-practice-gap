import torch
from torch import nn
from yarotsky import f

torch.set_default_dtype(torch.float64)


# Width-4 ReLU network with m hidden layers
def make_net(m: int) -> nn.Sequential:
    layers = [nn.Linear(1, 4), nn.ReLU()]
    for _ in range(m - 1):
        layers += [nn.Linear(4, 4), nn.ReLU()]
    layers.append(nn.Linear(4, 1))
    return nn.Sequential(*layers)



# Overwrite the random weights so the network computes f_m
def set_yarotsky_weights(net: nn.Sequential, m: int) -> None:
    linears = [layer for layer in net if isinstance(layer, nn.Linear)]
    with torch.no_grad():
        # First layer: x -> (relu(x), relu(x - 1/2), relu(x - 1), relu(x))
        linears[0].weight.copy_(torch.tensor([[1.0], [1.0], [1.0], [1.0]]))
        linears[0].bias.copy_(torch.tensor([0.0, -0.5, -1.0, 0.0]))

        # Middle layers: pieces of the next sawtooth, and the running total
        for k in range(1, m):
            c = 4.0**-k
            linears[k].weight.copy_(torch.tensor([
                [2.0, -4.0, 2.0, 0.0],
                [2.0, -4.0, 2.0, 0.0],
                [2.0, -4.0, 2.0, 0.0],
                [-2*c, 4*c, -2*c, 1.0],
            ]))
            linears[k].bias.copy_(torch.tensor([0.0, -0.5, -1.0, 0.0]))

        # Output layer: the final running total f_m
        c = 4.0**-m
        linears[m].weight.copy_(torch.tensor([[-2*c, 4*c, -2*c, 1.0]]))
        linears[m].bias.zero_()


# Check that the network matches the formula for m = 1..12
if __name__ == "__main__":
    x = torch.linspace(0, 1, 2**16 + 1)
    for m in range(1, 13):
        net = make_net(m)
        set_yarotsky_weights(net, m)
        with torch.no_grad():
            out = net(x.reshape(-1, 1)).flatten()
        print(m, (out - f(x, m)).abs().max().item())