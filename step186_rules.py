import sympy as sp

x = sp.Symbol('x')

print("product :", sp.diff(x**2 * x**3, x))
print("quotient:", sp.diff(x**3 / x, x))
print("chain   :", sp.diff((x**2 + 1)**3, x))
print("chain at x=1:", sp.diff((x**2 + 1)**3, x).subs(x, 1))