import numpy as np


def adam(grad_fn, w0, alpha, steps, beta1=0.9, beta2=0.999, eps=1e-8):
    w = np.array(w0, dtype=float)
    m = np.zeros_like(w)
    v = np.zeros_like(w)
    history = []

    for t in range(1, steps + 1):
        g = np.array(grad_fn(w), dtype=float)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        w = w - alpha * m_hat / (np.sqrt(v_hat) + eps)
        history.append(w.copy())

    return history


if __name__ == "__main__":
    # Edge cases: 0 steps -> empty history; starting at minimum [0.0] -> stays at 0.0 cleanly
    assert adam(lambda w: 2 * w, [2.0], alpha=0.1, steps=0) == []
    assert np.allclose(adam(lambda w: 2 * w, [0.0], alpha=0.1, steps=2), [[0.0], [0.0]])

    # Test 1: Match hand calculation (1.9000 and 1.8002)
    path_base = adam(lambda w: 2 * w, [2.0], alpha=0.1, steps=5)
    assert np.allclose([path_base[0][0], path_base[1][0]], [1.9000, 1.8002], atol=1e-4)

    # Test 2: Scale invariance (grad = 200w matches grad = 2w within 1e-6)
    path_scaled = adam(lambda w: 200 * w, [2.0], alpha=0.1, steps=5)
    max_diff = np.max(np.abs(np.array(path_base) - np.array(path_scaled)))
    assert max_diff < 1e-6

    print("Test 1 (paper match):", [round(float(x[0]), 4) for x in path_base[:2]])
    print("Test 2 (scale match):", [round(float(x[0]), 6) for x in path_scaled])
    print(f"Max diff between 2w and 200w over 5 steps: {max_diff:.2e}")