import numpy as np


def cce(Y, P, eps=1e-15):
  P = np.clip(P, eps, 1 - eps)
  row_losses = -np.sum(Y * np.log(P), axis=1)
  return np.mean(row_losses)


def hinge(y, score):
  row_losses = np.maximum(0.0, 1.0 - y * score)
  return np.mean(row_losses)


Y = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
P = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.3, 0.3, 0.4]])

y = np.array([1, -1, 1, -1])
score = np.array([2.0, -0.5, 0.3, 1.2])

print(f"CCE mean:   {cce(Y, P):.4f}")
print(f"Hinge mean: {hinge(y, score):.4f}")