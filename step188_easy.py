x = 1.0

# Forward pass
u = x**2 + 1.0
y = u**3
print(f"u: {u}, y: {y}")

# Backward pass
dy_dy = 1.0
local_dy_du = 3.0 * (u**2)
dy_du = dy_dy * local_dy_du

local_du_dx = 2.0 * x
dy_dx = dy_du * local_du_dx
print(f"dy_du: {dy_du}, dy_dx: {dy_dx}")

# Central-difference check
h = 1e-5
y_plus = ((x + h) ** 2 + 1.0) ** 3
y_minus = ((x - h) ** 2 + 1.0) ** 3
numeric_dy_dx = (y_plus - y_minus) / (2.0 * h)
print(f"numeric check: {numeric_dy_dx}")