import numpy as np

rng = np.random.default_rng(42)
flips = rng.random(10000) < 0.5      # True means heads

for n in [10, 100, 1000, 10000]:
    print(f"n = {n:>5}: share of heads = {flips[:n].mean():.4f}")