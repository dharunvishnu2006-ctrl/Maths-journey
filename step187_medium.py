import sympy as sp

x, y, z = sp.symbols('x y z')
f = x**2 * z + sp.exp(x * y) - 4 * z

# gradient components
gx = sp.diff(f, x)
gy = sp.diff(f, y)
gz = sp.diff(f, z)

# evaluate at (1, 0, 2)
ans_x = gx.subs(x, 1).subs(y, 0).subs(z, 2)
ans_y = gy.subs(x, 1).subs(y, 0).subs(z, 2)
ans_z = gz.subs(x, 1).subs(y, 0).subs(z, 2)

grad = [ans_x, ans_y, ans_z]
print(grad)