x = 2.0
y = 3.0

# Forward pass
m = x * y
f = m + x
print(f"m: {m}, f: {f}")

# Backward pass
df_df = 1.0

# Local gradient at m: df/dm = 1.0 (since f = m + x)
df_dm = df_df * 1.0

# Gradients for x (accumulated across both paths)
grad_x = 0.0
# Path 1: direct branch into f (f = m + x -> df/dx = 1.0)
grad_x += df_df * 1.0
# Path 2: branch through m (m = x * y -> dm/dx = y = 3.0)
grad_x += df_dm * y

# Gradient for y (single branch through m: dm/dy = x = 2.0)
grad_y = df_dm * x

print(f"df_dx: {grad_x}, df_dy: {grad_y}")

# Central-difference check
h = 1e-5

# Check wrt x
f_x_plus = (x + h) * y + (x + h)
f_x_minus = (x - h) * y + (x - h)
num_df_dx = (f_x_plus - f_x_minus) / (2.0 * h)

# Check wrt y
f_y_plus = x * (y + h) + x
f_y_minus = x * (y - h) + x
num_df_dy = (f_y_plus - f_y_minus) / (2.0 * h)

print(f"numeric df_dx: {num_df_dx}")
print(f"numeric df_dy: {num_df_dy}")