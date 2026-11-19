import numpy as np

def mse(y, y_hat):
    return np.mean((y - y_hat) ** 2)

def mae(y, y_hat):
    return np.mean(np.abs(y - y_hat))

def r2(y, y_hat):
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - (ss_res / ss_tot)

# 1. Paper cases for R^2
y_paper = np.array([2, 4, 6, 8])
y_hat_09 = np.array([3, 4, 5, 8])
y_hat_mean = np.array([5, 5, 5, 5])

print(f"Case 1 (0.9):  R2 = {r2(y_paper, y_hat_09):.2f}")
print(f"Case 2 (Mean): R2 = {r2(y_paper, y_hat_mean):.2f}")
print("-" * 35)

# 2. Outlier test
y_clean = np.array([10, 12, 14, 16])
y_hat   = np.array([11, 11, 15, 15])

mse_before = mse(y_clean, y_hat)
mae_before = mae(y_clean, y_hat)
print(f"Before outlier -> MSE: {mse_before:.1f}, MAE: {mae_before:.1f}")

y_outlier = np.array([10, 12, 14, 50])
mse_after = mse(y_outlier, y_hat)
mae_after = mae(y_outlier, y_hat)
print(f"After outlier  -> MSE: {mse_after:.1f}, MAE: {mae_after:.1f}")
print("-" * 35)

print(f"MSE growth factor: {mse_after / mse_before:.1f}x")
print(f"MAE growth factor: {mae_after / mae_before:.1f}x")