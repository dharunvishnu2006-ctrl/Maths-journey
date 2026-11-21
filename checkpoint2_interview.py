import numpy as np

z, y = -4.0, 1.0
a = 1 / (1 + np.exp(-z))
grad_mse = 2 * (a - y) * a * (1 - a)
grad_bce = a - y
print(
    f"MSE: {grad_mse:.4f} | BCE: {grad_bce:.4f} | Ratio:"
    f" {grad_bce/grad_mse:.1f}x"
)