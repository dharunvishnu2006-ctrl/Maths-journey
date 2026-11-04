import matplotlib.pyplot as plt
import numpy as np

# 1. Prepare data
km = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0])
y = np.array([16.0, 18.0, 23.0, 26.0, 27.0, 31.0, 34.0, 41.0])
X = np.column_stack([np.ones(len(y)), km])

# 2. Closed-form check
w_closed = np.linalg.inv(X.T @ X) @ X.T @ y


# 3. GD with early stopping
def gradient_descent(X, y, alpha, max_epochs=20000, tol=1e-9):
  n = X.shape[0]
  w = np.zeros(2)
  losses = []

  for epoch in range(max_epochs):
    pred = X @ w
    error = pred - y
    loss = (error**2).sum() / (2 * n)
    losses.append(loss)

    if epoch > 0 and (np.isnan(loss) or np.isinf(loss) or loss > losses[-2]):
      return w, losses, epoch

    if epoch > 0 and 0 < (losses[-2] - loss) < tol:
      return w, losses, epoch

    grad = X.T @ error / n
    w = w - alpha * grad

  return w, losses, max_epochs


# 4. Compare alphas: 0.001, 0.01, 0.1
alphas = [0.001, 0.01, 0.1]
curves = {}

for alpha in alphas:
  w_res, losses_res, stop_ep = gradient_descent(
      X, y, alpha=alpha, max_epochs=20000, tol=1e-9
  )
  curves[alpha] = losses_res
  print(f"alpha = {alpha:<5} | stopped epoch: {stop_ep}")

# 6. Predict for 3.2 km using alpha = 0.1
w_best, _, _ = gradient_descent(X, y, alpha=0.1, max_epochs=20000, tol=1e-9)
dist_target = 3.2
pred_3_2 = w_best[0] + w_best[1] * dist_target
print(f"Predicted delivery time for {dist_target} km: {pred_3_2:.2f} minutes")

# Step 4: Test divergence guard with alpha = 0.3
w_div, losses_div, stopped_ep_div = gradient_descent(
    X, y, alpha=0.3, max_epochs=20000, tol=1e-9
)
print("alpha = 0.3   | stopped epoch:", stopped_ep_div)
print("last two losses (alpha=0.3):", losses_div[-2:])

# Plotting at the very end so prints are not blocked
plt.figure(figsize=(8, 5))
for alpha in alphas:
  plt.plot(curves[alpha], label=f"alpha = {alpha}")

plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("Loss (log scale)")
plt.title("Loss Curves for Alphas")
plt.legend()
plt.grid(True)
plt.show()