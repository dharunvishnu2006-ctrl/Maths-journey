
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

# Coin simulation
n = 10000
flips = rng.random(n) < 0.5
share = np.cumsum(flips) / np.arange(1, n + 1)

plt.plot(share, label="Heads share")
plt.axhline(0.5, color="red", linestyle="--")
plt.xscale("log")
plt.xlabel("Number of flips")
plt.ylabel("Share of heads")
plt.legend()
plt.grid()
plt.show()

# Dice simulation
dice = rng.integers(1, 7, (100000, 2))
sums = dice.sum(axis=1)

for n in [100, 1000, 10000, 100000]:
    p = np.mean(sums[:n] == 7)
    print(f"n={n:6d}, P={p:.5f}, error={abs(p-1/6):.5f}")
