import numpy as np

y = np.array([10.0, 20.0, 30.0, 40.0])
y_hat = np.array([20.0, 10.0, 40.0, 30.0])

e = y - y_hat
mae = np.mean(np.abs(e))
print("e:", e)
print("MAE:", mae)