import sympy as sp

x, y = sp.symbols('x y')
f = x**2 * y + 3*y

df_dx = sp.diff(f, x)
df_dy = sp.diff(f, y)

print(df_dx)
print(df_dy)

# gradient vector: one partial derivative per variable
grad = [sp.diff(f, v) for v in (x, y)]
print(grad)

# evaluate the gradient at the point (2, 1)
point = {x: 2, y: 1}
grad_at = [g.subs(point) for g in grad]
print(grad_at)

# numeric check: nudge ONE input at a time (central difference)
f_num = sp.lambdify((x, y), f)
h = 1e-5
dx_num = (f_num(2 + h, 1) - f_num(2 - h, 1)) / (2 * h)
dy_num = (f_num(2, 1 + h) - f_num(2, 1 - h)) / (2 * h)
print(dx_num, dy_num)