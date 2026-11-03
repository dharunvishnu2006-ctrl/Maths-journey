import math

xs = [1, 2, 3, 4]
ys = [15, 30, 45, 60]


def compute_loss(w):
  return sum((w * x - y) ** 2 for x, y in zip(xs, ys)) / len(xs)


def compute_grad(w):
  return sum(2 * (w * x - y) * x for x, y in zip(xs, ys)) / len(xs)


# --- diagnose(history) ---
def diagnose(history):
  # 0. Guard against short histories
  if len(history) < 2:
    return "too slow"

  # 1. If history[-1] > history[0] * 1.5, or history has NaN/inf: return "diverged".
  if (
      math.isnan(history[-1])
      or math.isinf(history[-1])
      or (history[-1] > history[0] * 1.5)
  ):
    return "diverged"

  # 2. If history[-1] < 1.0 and abs(history[-1] - history[-2]) < 0.1: return "converged".
  if (history[-1] < 1.0) and (abs(history[-1] - history[-2]) < 0.1):
    return "converged"

  # 3. Otherwise: return "too slow".
  return "too slow"


print(diagnose([0.5]))