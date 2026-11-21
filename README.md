# Yarotsky's x² network: construction vs training

Yarotsky (2017) builds a ReLU network of width 4 and depth $m$ that approximates $x^2$ on $[0,1]$ with sup error $2^{-2m-2}$. Here I build that network by hand and compare it with the same architecture trained from random weights.

## Construction

With $\sigma(t) = \max(0,t)$ and the hat function $g(x) = 2\sigma(x) - 4\sigma(x - \tfrac12) + 2\sigma(x-1)$, let $g_s$ be $g$ composed $s$ times. Then

$$f_m(x) = x - \sum_{s=1}^m 4^{-s} g_s(x)$$

interpolates $x^2$ on the grid $k/2^m$, and $\|f_m - x^2\|_\infty = 2^{-2m-2}$. Each layer adds one sawtooth, so $f_m$ is a width-4 network with $m$ hidden layers.

## Code

- `sawtooth.py`: $g$, $g_s$ and a plot of $g_1, \dots, g_4$
- `yarotsky.py`: $f_m$ and a check of the error formula for $m \le 12$
- `network.py`: the network with hand-set weights; matches $f_m$ exactly
- `train.py`: trains the same architecture from random weights
- `plot.py`: the comparison figure

Training: 1000 uniform samples, MSE loss, Adam (lr $10^{-3}$), 20 000 steps, 5 seeds per depth, float64. Errors are measured in the sup norm on $2^{16}+1$ grid points.

## Results

![](figures/gap.png)

The trained error stops improving at around $5 \times 10^{-3}$ from $m = 5$, while the construction keeps gaining a factor of 4 per layer. At $m = 12$ that is $4 \times 10^{-3}$ against $1.5 \times 10^{-8}$.

For $m = 4$ and $m \ge 8$, most runs end with a constant output (sup error $\approx 0.67$). Some layer has all its ReLUs off on $[0,1]$, so the gradients vanish and training stalls. Running 10 times longer didn't change these runs.

Note: training uses MSE while the comparison uses the sup norm.

## Running

```
pip install -r requirements.txt
python train.py
python plot.py
```

## References

- D. Yarotsky, Error bounds for approximations with deep ReLU networks, Neural Networks (2017)
- B. Adcock, N. Dexter, The gap between theory and practice in function approximation with deep neural networks, SIAM J. Math. Data Sci. (2021)