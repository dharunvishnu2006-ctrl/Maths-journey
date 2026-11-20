import numpy as np

def bce(y, p, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

for p in [0.9, 0.5, 0.1, 0.01]:
    print(f"p={p}: BCE={bce(np.array([1]), np.array([p])):.4f}, SE={(1 - p) ** 2:.4f}")

print("p = 0 exactly:", bce(np.array([1]), np.array([0.0])))