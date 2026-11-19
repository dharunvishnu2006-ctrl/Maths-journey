import numpy as np

# 1. Define Ravi's 8 normal delivery rows plus the 9th procession outlier row (3 km, 150 min), and build design matrix X with bias column.
km_normal = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0])
min_normal = np.array([16.0, 18.0, 23.0, 26.0, 27.0, 31.0, 34.0, 41.0])

km = np.append(km_normal, 3.0)
y = np.append(min_normal, 150.0)

X_normal = np.column_stack([np.ones_like(km_normal), km_normal])
X = np.column_stack([np.ones_like(km), km])

# 2. Define metric helper r2(y, y_hat) with a zero-variance (SS_tot == 0) guard.
def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return 1.0 - (ss_res / ss_tot)

# 3. Define a single gradient descent function fit_gd(X, y, loss_kind, alpha, epochs, delta) supporting 'mse', 'mae', and 'huber'.
def fit_gd(X, y, loss_kind="mse", alpha=0.02, epochs=20000, delta=5.0):
    n, d = X.shape
    w = np.zeros(d, dtype=float)
    for _ in range(epochs):
        e = X @ w - y
        if loss_kind == "mse":
            grad = (X.T @ (2.0 * e)) / n
        elif loss_kind == "mae":
            grad = (X.T @ np.sign(e)) / n
        elif loss_kind == "huber":
            grad = (X.T @ np.clip(e, -delta, delta)) / n
        else:
            raise ValueError(f"Unknown loss_kind: {loss_kind}")
        w -= alpha * grad
    return w

# 4. Run edge-case checks (constant y in r2, extreme delta behavior in Huber, and e=0 gradient handling).
assert np.isnan(r2(np.array([7.0, 7.0, 7.0]), np.array([7.0, 7.0, 7.0]))), "r2 should return nan when SS_tot == 0"
w_huber_huge_delta = fit_gd(X_normal, min_normal, loss_kind="huber", alpha=0.04, epochs=5000, delta=1e6)
w_mse_clean = fit_gd(X_normal, min_normal, loss_kind="mse", alpha=0.02, epochs=5000)
assert np.allclose(w_huber_huge_delta, w_mse_clean, atol=1e-3), "Huber with huge delta should match MSE"
assert np.sign(0.0) == 0.0, "np.sign(0) must be 0"
print("Edge-case checks passed.\n")

# 5. Fit the linear model on all 9 rows three times: MSE (alpha=0.02), MAE (alpha=0.05), and Huber (alpha=0.05, delta=5.0) for 20,000 epochs.
configs = [
    ("MSE",   "mse",   0.02, 5.0),
    ("MAE",   "mae",   0.05, 5.0),
    ("Huber", "huber", 0.05, 5.0),
]

x_test_3_2 = np.array([1.0, 3.2])

# 6. For each fitted w, evaluate R^2 on the 8 normal rows only and compute the prediction for a 3.2 km delivery.
print(f"{'Loss':<8} | {'Fitted w [w0, w1]':<22} | {'R² (8 normal)':<15} | {'Pred (3.2 km)':<14}")
print("-" * 68)
for name, kind, alpha, delta in configs:
    w = fit_gd(X, y, loss_kind=kind, alpha=alpha, epochs=20000, delta=delta)
    r2_normal = r2(min_normal, X_normal @ w)
    pred_3_2 = x_test_3_2 @ w
    print(f"{name:<8} | [{w[0]:>7.3f}, {w[1]:>7.3f}]     | {r2_normal:>13.4f}   | {pred_3_2:>9.2f} min")

# 7. Print the comparison summary, the rule for choosing a loss by outlier profile, and the final recommendation for Ravi.
print("\n--- Rule for Choosing Loss by Outlier Profile ---")
print("1. Clean / Gaussian noise (no big outliers)  -> MSE   (fast, exact mean, smooth gradients everywhere)")
print("2. Heavy / frequent outliers, median target  -> MAE   (ignores outlier magnitude completely, nonsmooth at e=0)")
print("3. Mostly clean data + occasional spikes     -> Huber (smooth MSE behavior for normal noise, bounded MAE pull on spikes)")

print("\n--- Recommendation for Ravi ---")
print("Give Ravi Huber loss (delta = 5 min). Normal deliveries stay in the quadratic bowl for accurate, smooth optimization,")
print("while rare traffic/procession spikes are capped at a 5-minute gradient pull instead of wrecking his everyday ETAs.")