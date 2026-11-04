import numpy as np

# Data setup
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y = np.array([3.0, 5.0, 7.0, 9.0, 11.0])
X = np.column_stack([np.ones(len(y)), x])


def fit_gd_fixed(X, y, alpha=0.05, epochs=2000):
  n = X.shape[0]
  w = np.zeros(X.shape[1])
  losses = []
  for _ in range(epochs):
    pred = X @ w
    error = pred - y  # Fixed line
    loss = (error**2).sum() / (2 * n)
    losses.append(loss)
    grad = X.T @ error / n
    w = w - alpha * grad
  return w, losses


w_fixed, losses_fixed = fit_gd_fixed(X, y, alpha=0.05, epochs=2000)

print(f"First 3 losses: {[round(l, 4) for l in losses_fixed[:3]]}")
print(f"Last 3 losses:  {[round(l, 8) for l in losses_fixed[-3:]]}")
print(f"Final w:        {np.round(w_fixed, 4)}")