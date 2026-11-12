import numpy as np
import time

# 1. Generate 1,000 noisy student records (seed=7) with hours in [0, 6], marks = 35 + 8*hours + N(0, 25), and build X with a bias column of 1s.
rng = np.random.default_rng(7)
hours = rng.uniform(0, 6, size=1000)
noise = rng.normal(0, 5, size=1000)
y = 35 + 8 * hours + noise
X = np.column_stack([np.ones_like(hours), hours])


# 2. Define minibatch_sgd to shuffle indices per epoch, update w over batches of size b (scaling by len(idx)), count total updates, log full-data loss, and save epoch 0 and 1 first-batch indices.
def minibatch_sgd(X, y, alpha, epochs, b, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    w = np.zeros(X.shape[1])
    losses = []
    updates = 0
    first_batch_ep0 = None
    first_batch_ep1 = None

    for epoch in range(epochs):
        perm = rng.permutation(n)
        for start in range(0, n, b):
            idx = perm[start:start + b]
            if start == 0:
                if epoch == 0:
                    first_batch_ep0 = idx.copy()
                elif epoch == 1:
                    first_batch_ep1 = idx.copy()
            Xb, yb = X[idx], y[idx]
            error = Xb @ w - yb
            grad = Xb.T @ error / len(idx)
            w = w - alpha * grad
            updates += 1
        full_error = X @ w - y
        losses.append((full_error ** 2).sum() / (2 * n))

    return w, losses, updates, first_batch_ep0, first_batch_ep1


# 3. Verify shuffling by comparing epoch 0 vs. epoch 1 first-batch indices to confirm they differ.
w, losses, updates, ep0_idx, ep1_idx = minibatch_sgd(X, y, alpha=0.01, epochs=20, b=64)
print("Total updates:", updates)
print("Epoch 0 first batch (first 5):", ep0_idx[:5])
print("Epoch 1 first batch (first 5):", ep1_idx[:5])
print("Indices differ:", not np.array_equal(ep0_idx, ep1_idx))

# 4. Loop over b in [1, 4, 16, 64, 256, 1000] (alpha=0.01, epochs=20), timing each run with time.perf_counter() and recording final loss, elapsed seconds, and total updates.
for b in [1, 4, 16, 64, 256, 1000]:
    t0 = time.perf_counter()
    w, losses, updates, _, _ = minibatch_sgd(X, y, alpha=0.01, epochs=20, b=b)
    elapsed = time.perf_counter() - t0
    print(
        f"b: {b:4d} | final loss: {losses[-1]:10.4f} | time: {elapsed:.5f}s |"
        f" updates: {updates}"
    )
# 5. Compare results and recommend the batch size that reaches near-minimum loss fastest without high-variance jitter or underfitting.
# Pattern: Mini-batch Stochastic Gradient Descent (empirical hyperparameter sweep).