import numpy as np


def adam(grad_fn, w0, alpha, steps, beta1=0.9, beta2=0.999, eps=1e-8):
    w = np.array(w0, dtype=float)
    m = np.zeros_like(w)
    v = np.zeros_like(w)
    history = []

    for t in range(1, steps + 1):
        g = grad_fn(w)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g ** 2
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        w = w - alpha * m_hat / (np.sqrt(v_hat) + eps)
        history.append(w.copy())

    return history


def grad(w):
    return np.array([w[0], 50.0 * w[1]])


# Test 1: First 3 steps from [10, 1]
res1 = adam(grad, [10.0, 1.0], alpha=0.1, steps=3)
expected = [[9.9, 0.9], [9.8, 0.8004], [9.7001, 0.7016]]
print("Test 1:", np.round(res1, 4).tolist())
assert np.allclose(res1, expected, atol=1e-4)

# Test 2: Edge case at minimum [0, 0]
res2 = adam(grad, [0.0, 0.0], alpha=0.1, steps=3)
print("Test 2:", res2[-1].tolist())
assert np.allclose(res2[-1], [0.0, 0.0])