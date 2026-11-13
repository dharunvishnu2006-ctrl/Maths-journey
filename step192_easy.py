import numpy as np

km = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0])
y = np.array([16.0, 18.0, 23.0, 26.0, 27.0, 31.0, 34.0, 41.0])
X = np.column_stack([np.ones_like(km), km])

w_closed = np.linalg.solve(X.T @ X, X.T @ y)
n = len(y)
min_loss = (1 / (2 * n)) * np.sum((X @ w_closed - y) ** 2)

def momentum_gd(X, y, alpha, beta, epochs):
    w = np.zeros(X.shape[1])
    v = np.zeros_like(w)
    losses = []
    for _ in range(epochs):
        err = X @ w - y
        losses.append((1 / (2 * n)) * np.sum(err ** 2))
        grad = (1 / n) * (X.T @ err)
        v = beta * v - alpha * grad
        w = w + v
    return w, losses

w_gd, losses_gd = momentum_gd(X, y, alpha=0.1, beta=0.0, epochs=2000)
w_m, losses_m = momentum_gd(X, y, alpha=0.1, beta=0.9, epochs=2000)

target = min_loss + 1e-6
ep_gd = next(i + 1 for i, l in enumerate(losses_gd) if l <= target)
ep_m = next(i + 1 for i, l in enumerate(losses_m) if l <= target)

print(f"min_loss: {min_loss:.8f} | w_closed: {w_closed.round(4)}")
print(f"Plain GD (beta=0.0): epoch {ep_gd} | w: {w_gd.round(4)}")
print(f"Momentum (beta=0.9): epoch {ep_m} | w: {w_m.round(4)}")
print(f"Speedup: {ep_gd / ep_m:.2f}x")