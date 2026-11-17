import numpy as np


def adam(grad_fn, w0, alpha, steps, beta1=0.9, beta2=0.999, eps=1e-8, bias_correction=True):
    w = np.array(w0, dtype=float)
    m = np.zeros_like(w)
    v = np.zeros_like(w)
    history = []

    for t in range(1, steps + 1):
        g = grad_fn(w)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g ** 2
        if bias_correction:
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
        else:
            m_hat = m
            v_hat = v
        w = w - alpha * m_hat / (np.sqrt(v_hat) + eps)
        history.append(w.copy())

    return history


def grad(w):
    return np.array([w[0], 50.0 * w[1]])


print("With bias correction:")
for t, w in enumerate(adam(grad, [10.0, 1.0], alpha=0.1, steps=5, bias_correction=True), 1):
    print(f"t={t}: w = {np.round(w, 4)}")

print("\nWithout bias correction:")
for t, w in enumerate(adam(grad, [10.0, 1.0], alpha=0.1, steps=5, bias_correction=False), 1):
    print(f"t={t}: w = {np.round(w, 4)}")