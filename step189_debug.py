xs = [1, 2, 3, 4]
ys = [25, 40, 55, 70]
n = len(xs)


def loss(w, b):
  return sum((w * x + b - y) ** 2 for x, y in zip(xs, ys)) / n


w, b, alpha = 0.0, 0.0, 0.1
for step in range(300):
  gw = sum(2 * (w * x + b - y) * x for x, y in zip(xs, ys)) / n
  gb = sum(2 * (w * x + b - y) for x, y in zip(xs, ys)) / n
  w = w - alpha * gw
  b = b - alpha * gb

print(round(w, 4), round(b, 4), loss(w, b))