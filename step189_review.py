# --- Understand & Plan ---
# L(w, b) = mean((w*x_i + b - y_i)^2)
# dL/dw = mean(2 * (w*x_i + b - y_i) * x_i)
# dL/db = mean(2 * (w*x_i + b - y_i))
# 1. Init w=0, b=0, alpha=0.1, steps=300.
# 2. Each step: compute both gradients with old w, b; update simultaneously.
# 3. Track losses and check if strictly decreasing every step.

xs = [1, 2, 3, 4]
ys = [25, 40, 55, 70]
n = len(xs)

w, b, alpha, steps = 0.0, 0.0, 0.1, 300
history = [sum((w * x + b - y) ** 2 for x, y in zip(xs, ys)) / n]

for _ in range(steps):
  gw = sum(2 * (w * x + b - y) * x for x, y in zip(xs, ys)) / n
  gb = sum(2 * (w * x + b - y) for x, y in zip(xs, ys)) / n
  w -= alpha * gw
  b -= alpha * gb
  history.append(sum((w * x + b - y) ** 2 for x, y in zip(xs, ys)) / n)

decreased_every_step = all(
    history[i] < history[i - 1] for i in range(1, len(history))
)

print(f"Final w: {w:.4f}, b: {b:.4f}, L: {history[-1]:.4e}")
print(f"Decreased every step: {decreased_every_step}")