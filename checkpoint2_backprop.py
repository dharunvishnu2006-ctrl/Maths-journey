import numpy as np

x = 2.0
w = 0.5
b = 0.1
alpha = 0.1
h = 1e-5

def loss(w, b):
    z = w * x + b
    a = 1 / (1 + np.exp(-z))
    return -np.log(a)

# Hand gradients
z = w * x + b
a = 1 / (1 + np.exp(-z))

dL_dz = a - 1
hand_dw = dL_dz * x
hand_db = dL_dz

# Numeric gradients
numeric_dw = (loss(w+h, b) - loss(w-h, b)) / (2*h)
numeric_db = (loss(w, b+h) - loss(w, b-h)) / (2*h)

# Comparison
differences = np.abs(
    np.array([hand_dw, hand_db]) -
    np.array([numeric_dw, numeric_db])
)

print("Hand gradients:", hand_dw, hand_db)
print("Numeric gradients:", numeric_dw, numeric_db)
print("Largest difference:", np.max(differences))

# Weight update
print("New w:", w - alpha * hand_dw)
print("New b:", b - alpha * hand_db)