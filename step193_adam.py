import numpy as np

def grad(w):
    return np.array([w[0], 50.0 * w[1]])

alpha, beta1, beta2, eps = 0.1, 0.9, 0.999, 1e-8
w = np.array([10.0, 1.0])
m = np.zeros(2)
v = np.zeros(2)

for t in range(1, 4):
    g = grad(w)
    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * g ** 2
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    w = w - alpha * m_hat / (np.sqrt(v_hat) + eps)
    print(f"t={t}: w = {np.round(w, 4)}")