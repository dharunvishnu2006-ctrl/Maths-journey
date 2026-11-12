import numpy as np
import matplotlib.pyplot as plt


def minibatch_sgd(X, y, alpha, epochs, b, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    w = np.zeros(X.shape[1])
    losses = []
    for epoch in range(epochs):
        perm = rng.permutation(n)
        for start in range(0, n, b):
            idx = perm[start:start + b]
            Xb, yb = X[idx], y[idx]
            error = Xb @ w - yb
            grad = Xb.T @ error / len(idx)
            w = w - alpha * grad
        full_error = X @ w - y
        losses.append((full_error ** 2).sum() / (2 * n))
    return w, losses


rng = np.random.default_rng(1)
x = rng.uniform(0, 4, size=200)
noise = rng.normal(0, 1, size=200)
y = 1 + 2 * x + noise
X = np.column_stack([np.ones_like(x), x])

plt.figure(figsize=(8, 5))
for b in [1, 16, 200]:
    w, losses = minibatch_sgd(X, y, alpha=0.05, epochs=30, b=b)
    print(f"b: {b} | last loss: {losses[-1]}")
    plt.plot(losses, label=f"b = {b}")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curves: SGD (b=1) vs Mini-batch (b=16) vs Batch (b=200)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("step191_medium.png")
