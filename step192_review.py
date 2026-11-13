import numpy as np

# 1. Define grad_fn(w) returning np.array([w[0], 4.0 * w[1]], dtype=float).
def grad_fn(w):
    return np.array([w[0], 4.0 * w[1]], dtype=float)

# 2. Define optimize(grad_fn, w0, alpha, beta, steps, nesterov=False).
def optimize(grad_fn, w0, alpha, beta, steps, nesterov=False):
    w = np.array(w0, dtype=float)
    v = np.zeros_like(w)
    history = []
    for _ in range(steps):
        g = grad_fn(w + beta * v) if nesterov else grad_fn(w)
        v = beta * v - alpha * g
        w = w + v
        history.append(w.copy())
    return history

# 3. Test 2 first (edge case): beta=0 gives identical paths for momentum, Nesterov, and plain GD.
w_gd = np.array([2.0, 1.0])
gd_history = []
for _ in range(5):
    w_gd = w_gd - 0.1 * grad_fn(w_gd)
    gd_history.append(w_gd.copy())

mom_zero = optimize(grad_fn, [2.0, 1.0], alpha=0.1, beta=0.0, steps=5, nesterov=False)
nes_zero = optimize(grad_fn, [2.0, 1.0], alpha=0.1, beta=0.0, steps=5, nesterov=True)

assert np.allclose(mom_zero, gd_history), "Momentum (beta=0) failed GD match"
assert np.allclose(nes_zero, gd_history), "Nesterov (beta=0) failed GD match"
print("Test 2 (edge case beta=0) PASSED:", [w.round(4).tolist() for w in mom_zero])

# 4. Test 1: verify w0=[2, 1], alpha=0.1, beta=0.5 matches paper steps [1.8, 0.6] and [1.52, 0.16].
paper_history = optimize(grad_fn, [2.0, 1.0], alpha=0.1, beta=0.5, steps=2, nesterov=False)
expected = [[1.8, 0.6], [1.52, 0.16]]
assert np.allclose(paper_history, expected), f"Expected {expected}, got {paper_history}"
print("Test 1 (paper match) PASSED:", [w.round(4).tolist() for w in paper_history])