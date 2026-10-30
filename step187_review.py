# Understand: We need to find the gradient of f(x, y, z) = x^2*y - y*z^3 + 4*x
# at (1, 2, -1) both symbolically with sympy and numerically using central differences.
# Plan:
# 1. Store the point once at the top.
# 2. Define the symbols and the function f.
# 3. Find the symbolic gradient using a loop over the variables.
# 4. Evaluate the symbolic gradient at the point.
# 5. Turn f into a python function and compute the numerical gradient with a loop.
# 6. Compare both gradients and check if they agree within 1e-6.

import sympy as sp

# 1. Store the point once at the top
point_values = (1, 2, -1)

# 2. Define symbols and function
x, y, z = sp.symbols('x y z')
variables = (x, y, z)
f = x**2 * y - y * z**3 + 4 * x

# 3. Build the full gradient with sp.diff in a loop
grad = []
for v in variables:
    grad.append(sp.diff(f, v))

# 4. Evaluate symbolic gradient at the point
point = {x: point_values[0], y: point_values[1], z: point_values[2]}
grad_at = []
for g in grad:
    grad_at.append(float(g.subs(point)))

# 5. Central-difference gradient in a loop (h = 1e-5)
f_num = sp.lambdify((x, y, z), f)
h = 1e-5

grad_num = []
pt_list = list(point_values)

for i in range(len(pt_list)):
    # Create forward and backward points by nudging just one variable
    fwd = list(pt_list)
    bwd = list(pt_list)
    
    fwd[i] = fwd[i] + h
    bwd[i] = bwd[i] - h
    
    # Central difference formula: (f(x+h) - f(x-h)) / (2*h)
    diff = (f_num(fwd[0], fwd[1], fwd[2]) - f_num(bwd[0], bwd[1], bwd[2])) / (2 * h)
    grad_num.append(diff)

# 6. Print both gradients and check agreement
print("Analytical gradient:", grad_at)
print("Numerical gradient: ", grad_num)

# Check if every component agrees within 1e-6
agrees = True
for a, n in zip(grad_at, grad_num):
    if abs(a - n) >= 1e-6:
        agrees = False
        break

print("All components agree within 1e-6:", agrees)