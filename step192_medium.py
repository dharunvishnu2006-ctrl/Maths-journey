import numpy as np

km = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0])
y = np.array([16.0, 18.0, 23.0, 26.0, 27.0, 31.0, 34.0, 41.0])
X = np.column_stack([np.ones_like(km), km])

w_closed = np.linalg.solve(X.T @ X, X.T @ y)
min_loss = (1 / (2 * len(y))) * np.sum((X @ w_closed - y) ** 2)

def momentum_gd(X, y, alpha, beta, epochs):
    n, d = X.shape
    w = np.zeros(d)
    v = np.zeros(d)
    losses = []
    for _ in range(epochs):
        err = X @ w - y
        losses.append((1 / (2 * n)) * np.sum(err ** 2))
        grad = (1 / n) * (X.T @ err)
        v = beta * v - alpha * grad
        w = w + v
    return w, losses

target = min_loss + 1e-6
for beta in [0, 0.5, 0.7, 0.9, 0.99]:
    w, losses = momentum_gd(X, y, alpha=0.1, beta=beta, epochs=2000)
    ep = next((i + 1 for i, l in enumerate(losses) if l <= target), None)
    print(f"beta = {beta:<4}: target epoch = {str(ep):<6} | final loss = {losses[-1]:.8f}")