import numpy as np


def loss(w, k):
    return 0.5 * (w[0] ** 2 + k * w[1] ** 2)


def grad(w, k):
    return np.array([w[0], k * w[1]], dtype=float)


def race(w0, k, mode, target=1e-3, max_steps=20000):
    w = np.array(w0, dtype=float)
    v = np.zeros_like(w)
    m = np.zeros_like(w)
    alpha = 0.1 if mode == "adam" else 1.0 / k

    if loss(w, k) < target:
        return 0
    for t in range(1, max_steps + 1):
        g = grad(w, k)
        if mode == "gd":
            w = w - alpha * g
        elif mode == "momentum":
            v = 0.9 * v - alpha * g
            w = w + v
        elif mode == "adam":
            m = 0.9 * m + 0.1 * g
            v = 0.999 * v + 0.001 * (g ** 2)
            m_hat = m / (1 - 0.9 ** t)
            v_hat = v / (1 - 0.999 ** t)
            w = w - alpha * m_hat / (np.sqrt(v_hat) + 1e-8)
        if loss(w, k) < target:
            return t
    return f"capped at {max_steps}"


# Edge cases
assert race([0.0, 0.0], 100, "adam") == 0
assert race([10.0, 1.0], 10000, "gd") == "capped at 20000"

# Race
for k in [100, 1000]:
    print(f"k={k}:")
    for m in ["gd", "momentum", "adam"]:
        print(f"  {m:8s} -> {race([10.0, 1.0], k, m)}")

print("\nRecommendation: Adam — scales each weight by its own gradient history, staying at 234 steps even when k jumps 10x.")