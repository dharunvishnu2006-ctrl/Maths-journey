def loss(w):
  return (2 * w - 8) ** 2


def grad(w):
  return 8 * (w - 4)


alphas = [0.125, 0.2, 0.26]

for alpha in alphas:
  w = 0.0
  for step in range(10):
    w = w - alpha * grad(w)
  L = loss(w)
  print(f"alpha = {alpha:5.3f} -> final w = {w:8.4f}, L = {L:8.4f}")