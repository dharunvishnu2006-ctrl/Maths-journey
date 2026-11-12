import numpy as np

X = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0]])
y = np.array([3.0, 5.0, 7.0, 9.0])


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


for b in [4, 1]:
    w, losses = minibatch_sgd(X, y, alpha=0.1, epochs=50, b=b)
    print("b:", b)
    print("final w:", w)
    print("last loss:", losses[-1])