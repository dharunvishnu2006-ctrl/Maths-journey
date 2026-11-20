import numpy as np

def bce(y, p, eps=1e-15):
    p = np.clip(p, eps, 1 - eps)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

def mse(y, p):
    return np.mean((y - p) ** 2)

def accuracy(y, p):
    preds = (p >= 0.5).astype(int)
    return np.mean(preds == y)

y = np.array([1, 0, 1, 1, 0])

models = {
    "A (good)": np.array([0.9, 0.1, 0.8, 0.7, 0.2]),
    "B (timid)": np.array([0.6, 0.4, 0.6, 0.6, 0.4]),
    "C (one confident blunder)": np.array([0.9, 0.1, 0.8, 0.01, 0.2]),
}

for name, p in models.items():
    print(f"{name:26s} | BCE: {bce(y, p):.4f} | MSE: {mse(y, p):.4f} | Acc: {accuracy(y, p):.2f}")
    