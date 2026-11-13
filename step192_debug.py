import numpy as np

def grad_fn(w):
    return np.array([w[0], 4.0 * w[1]], dtype=float)

def momentum(grad_fn, w0, alpha, beta, steps):
    w = np.array(w0, dtype=float)
    v = np.zeros_like(w)
    history = []
    for _ in range(steps):
        v = beta * v - alpha * grad_fn(w)
        w = w + v
        history.append(w.copy())
    return history

history = momentum(grad_fn, [2, 1], alpha=0.1, beta=0.5, steps=2)
print([w.round(4).tolist() for w in history])