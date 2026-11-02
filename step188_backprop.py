# forward pass: left to right
x = 2.0
u = 3*x + 1          # inner stage
y = u**2             # outer stage

# backward pass: right to left, starting from 1 at the output
dy_dy = 1.0
dy_du = dy_dy * (2*u)    # local rate of y = u**2
dy_dx = dy_du * 3        # local rate of u = 3x + 1

print(u, y)
print(dy_du, dy_dx)

# numeric check: central difference on the whole function
def f(x):
    return (3*x + 1)**2

h = 1e-5
print((f(x + h) - f(x - h)) / (2*h))

# the w, b node: forward pass
x, t = 2.0, 5.0
w, b = 1.0, 1.0
z = w*x + b
e = z - t
L = e**2

# backward pass: right to left
dL_dL = 1.0
dL_de = dL_dL * (2*e)
dL_dz = dL_de * 1
dL_dw = dL_dz * x
dL_db = dL_dz * 1
print(L, dL_dw, dL_db)

# accumulation: y = x * x at x = 3
x2 = 3.0
grad = 0.0
grad += x2      # path through a: dy/da = b = x2
grad += x2      # path through b: dy/db = a = x2
print(grad)

# the bug: overwriting instead of adding
bug = 0.0
bug = x2
bug = x2
print(bug)