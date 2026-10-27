import sympy as sp

x = sp.Symbol('x')
f = (x**2 + 1) * (x - 3)
g = x**2 / (x + 1)

print("f'    =", sp.expand(sp.diff(f, x)))
print("f'(2) =", sp.diff(f, x).subs(x, 2))
print("g'    =", sp.diff(g, x))
print("g'(1) =", sp.diff(g, x).subs(x, 1))