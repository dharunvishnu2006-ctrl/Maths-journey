import sympy as sp

x, y = sp.symbols('x y')
f = 3*x**3 + 5*x*y**2 - 7

df_dx = sp.diff(f, x)
print("df/dx =", df_dx)