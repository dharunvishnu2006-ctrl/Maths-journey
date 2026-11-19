import numpy as np

def mse(y, y_hat):
    return np.mean((y - y_hat) ** 2)

def mae(y, y_hat):
    return np.mean(np.abs(y - y_hat))

def huber(y, y_hat, delta=1.0):
    e = np.abs(y - y_hat)
    small = 0.5 * e ** 2
    large = delta * (e - 0.5 * delta)
    return np.mean(np.where(e <= delta, small, large))

y = np.array([10.0, 12.0, 14.0, 50.0])
y_hat = np.array([11.0, 11.0, 15.0, 14.0])
print("MSE:", mse(y, y_hat))
print("MAE:", mae(y, y_hat))
print("Huber (delta=2):", huber(y, y_hat, delta=2.0))
print("Huber (delta=100):", huber(y, y_hat, delta=100.0))