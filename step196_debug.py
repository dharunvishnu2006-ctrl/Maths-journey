import numpy as np

rng = np.random.default_rng(0)
dice = rng.integers(1, 7, size=(100000, 2))
sums = dice.sum(axis=1)

print("P(at least one 6):",
      np.mean(np.any(dice == 6, axis=1)))
print("P(sum = 12):", np.mean(sums == 12))
print("P(sum = 7):", np.mean(sums == 7))