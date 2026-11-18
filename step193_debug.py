import numpy as np


def adam(grad_fn, w0, alpha, steps, beta1=0.9, beta2=0.999, eps=1e-8):
    w = np.array(w0, dtype=float)
    m = np.zeros_like(w)
    v = np.zeros_like(w)
    history = []
    for t in range(1, steps + 1):
        g = np.array(grad_fn(w), dtype=float)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g ** 2
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        w = w - alpha * m_hat / (np.sqrt(v_hat) + eps)
        history.append(w.copy())
    return history


w1_2w = adam(lambda w: 2 * w, [2.0], alpha=0.1, steps=1)[0][0]
w1_200w = adam(lambda w: 200 * w, [2.0], alpha=0.1, steps=1)[0][0]
print(f"grad = 2w   -> step 1: {w1_2w:.4f}")
print(f"grad = 200w -> step 1: {w1_200w:.4f}")