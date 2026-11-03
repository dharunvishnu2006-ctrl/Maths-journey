def loss(w):
  return (2 * w - 8) ** 2

def grad(w):
  return 4 * (2 * w - 8)

w = 0.0
alpha = 0.05

for step in range(1, 11):
  g = grad(w)
  w = w - alpha * g
  L = loss(w)
  print(f"step {step:2d}: w = {w:.4f}, L = {L:.4f}")