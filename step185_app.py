import sympy as sp

x = sp.Symbol('x')

def compare(expr, x0, h=0.001):
    # 1. Guard against division by zero: ensure h != 0
    if h == 0:
        raise ValueError("Step size h cannot be 0 (division by zero).")

    # 2. Use the symbol x for everything. Ensure constants work with .subs()
    if not isinstance(expr, sp.Basic):
        expr = sp.sympify(expr)

    # 3. Compute exact symbolic derivative using sp.diff(expr, x)
    df = sp.diff(expr, x)

    # 4. Substitute x0 into the symbolic derivative using .subs() to get exact numerical value
    exact_val = float(df.subs(x, x0))

    # 5. Evaluate expr at (x0 + h) and at x0 using .subs()
    f_x0_plus_h = float(expr.subs(x, x0 + h))
    f_x0 = float(expr.subs(x, x0))

    # 6. Compute numeric derivative: (expr(x0 + h) - expr(x0)) / h
    numeric_val = (f_x0_plus_h - f_x0) / h

    # 7. Compute the absolute difference (error): abs(exact_val - numeric_val)
    error = abs(exact_val - numeric_val)

    # 8. Print exact value, numeric approximation, and the absolute error
    print(f"Expression: {expr} at x0 = {x0}, h = {h}")
    print(f"  Exact derivative:   {exact_val:.6f}")
    print(f"  Numeric derivative: {numeric_val:.6f}")
    print(f"  Absolute error:     {error:.6e}\n")

# --- The 4 Task Tests with h = 0.001 ---
compare(x**2, x0=3.0, h=0.001)
compare(x**3, x0=2.0, h=0.001)
compare(sp.sin(x), x0=0.0, h=0.001)
compare(sp.Integer(7), x0=5.0, h=0.001)