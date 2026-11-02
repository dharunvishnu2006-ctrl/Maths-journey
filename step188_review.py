# Plan: z = w*x, d = z - t, L = d^2 + w^2.
# Backward: grad_w = 2*d*x (via fit) + 2*w (via reg)

def loss(w, x, t):
    return (w * x - t) ** 2 + w ** 2

def backprop(w, x, t):
    # 1. Forward
    z, d = w * x, (w * x) - t
    L = d**2 + w**2
    # 2. Backward (path accumulation)
    grad_w = 0.0
    grad_w += (2.0 * d) * x   # Path 1: data fit
    grad_w += 2.0 * w         # Path 2: regularization
    return L, grad_w

def check(w, x, t, label):
    h = 1e-5
    L, dw = backprop(w, x, t)
    num = (loss(w + h, x, t) - loss(w - h, x, t)) / (2 * h)
    print(f"{label} | L: {L:.2f} | dw: {dw:.2f} | num: {num:.2f} | pass: {abs(dw - num) < 1e-7}")

# Run 1: Standard
check(0.5, 3.0, 2.0, "Standard")

# Run 2: Edge-case (w=0 tests if Path 2 vanishes without zeroing Path 1; x=0 tests Path 1)
check(0.0, 0.0, 1.0, "Edge-case")