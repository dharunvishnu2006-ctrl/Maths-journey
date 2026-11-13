import matplotlib.pyplot as plt

def grad(w):
    return 2 * (w - 10)

alpha, beta = 0.05, 0.9

w_gd = 0.0
w_m, v = 0.0, 0.0
ws_gd, ws_m = [], []

for step in range(1, 61):
    w_gd = w_gd - alpha * grad(w_gd)
    v = beta * v - alpha * grad(w_m)
    w_m = w_m + v
    ws_gd.append(w_gd)
    ws_m.append(w_m)

print(f"Final w (plain GD): {w_gd:.4f}")
print(f"Final w (momentum): {w_m:.4f}")

plt.figure(figsize=(8, 4.5))
plt.plot(range(1, 61), ws_gd, label="Plain GD", linewidth=2)
plt.plot(range(1, 61), ws_m, label="Momentum (β = 0.9)", linewidth=2)
plt.axhline(10, color="gray", linestyle="--", label="Target minimum (w = 10)")
plt.xlabel("Step")
plt.ylabel("w")
plt.title("Plain GD vs. Momentum over 60 Steps")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("piece2_chart.png", dpi=150)