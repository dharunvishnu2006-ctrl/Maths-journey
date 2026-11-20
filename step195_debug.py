import numpy as np


def bce(y, p, eps=1e-15):
    p = np.clip(p, eps, 1 - eps)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


y = np.array([1, 0])
p_good = np.array([0.9, 0.1])
p_lazy = np.array([0.9, 0.9])

print("Good model BCE:", round(bce(y, p_good), 4))
print("Lazy model BCE:", round(bce(y, p_lazy), 4))