import math

def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))

def loss(w, b, x, t):
    return (sigmoid(w * x + b) - t) ** 2

def backprop(w, b, x, t):
    # 1. Forward pass
    z = w * x + b
    a = sigmoid(z)
    
    # 2. Backward pass (one line per arrow)
    # 2a. Step back to a
    dL_da = 2.0 * (a - t)
    # 2b. Step back to z
    da_dz = a * (1.0 - a)
    dL_dz = dL_da * da_dz
    # 2c. Step back to w
    dL_dw = dL_dz * x
    # 2d. Step back to b
    dL_db = dL_dz * 1.0
    
    return dL_dw, dL_db

# Test values
w = 0.5
b = -1.0
x = 3.0
t = 1.0
h = 1e-5

dw, db = backprop(w, b, x, t)

# Numerical gradient check
for name, grad in [('w', dw), ('b', db)]:
    if name == 'w':
        num = (loss(w + h, b, x, t) - loss(w - h, b, x, t)) / (2.0 * h)
    else:
        num = (loss(w, b + h, x, t) - loss(w, b - h, x, t)) / (2.0 * h)
    diff = abs(grad - num)
    print(f"{name}: {grad:.6f} | num: {num:.6f} | diff: {diff:.2e} | pass: {diff < 1e-7}")