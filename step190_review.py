import numpy as np

# 1. Prepare data
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y = np.array([3.0, 5.0, 7.0, 9.0, 11.0])
X = np.column_stack([np.ones(len(y)), x])

# 2. fit_gd: plain gradient descent returning w and losses
def fit_gd(X, y, alpha, epochs):
  n = X.shape[0]
  w = np.zeros(X.shape[1])
  losses = []

  for _ in range(epochs):
    pred = X @ w
    error = pred - y
    loss = (error**2).sum() / (2 * n)
    losses.append(loss)

    grad = (X.T @ error) / n
    w = w - alpha * grad

  return w, losses

# 3. epochs_to_reach: returns first epoch where loss <= target, else -1
def epochs_to_reach(losses, target):
  for epoch, loss in enumerate(losses):
    if loss <= target:
      return epoch
  return -1

# 4. Run training
alpha = 0.05
epochs = 2000
w_final, losses = fit_gd(X, y, alpha, epochs)

# 5. Evaluate target (1% of initial loss)
target = 0.01 * losses[0]
reach_epoch = epochs_to_reach(losses, target)

# 6. Print outputs
print("Final w:           ", w_final)
print("Initial loss:      ", losses[0])
print("Target loss (1%):  ", target)
print("Epochs to reach:   ", reach_epoch)
print("Edge case 1 (empty list):", epochs_to_reach([], 1.0))
print("Edge case 2 (target 0.0):", epochs_to_reach(losses, 0.0))