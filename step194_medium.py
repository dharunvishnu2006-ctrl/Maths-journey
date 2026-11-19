import numpy as np
import matplotlib.pyplot as plt

def mse(y, y_hat):
    return np.mean((y - y_hat) ** 2)

def mae(y, y_hat):
    return np.mean(np.abs(y - y_hat))

def r2(y, y_hat):
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return 1 - (ss_res / ss_tot)

def huber(e, delta=1.0):
    abs_e = np.abs(e)
    return np.where(
        abs_e <= delta,
        0.5 * (e ** 2),
        delta * (abs_e - 0.5 * delta)
    )

# 1. Per-error penalty curves from -4 to 4
e = np.linspace(-4, 4, 401)
delta = 1.0

half_mse_curve = 0.5 * (e ** 2)
mae_curve = np.abs(e)
huber_curve = huber(e, delta=delta)

# 2. Check both Huber branches at e = delta = 1
e_switch = 1.0
quad_branch = 0.5 * (e_switch ** 2)
lin_branch = delta * (abs(e_switch) - 0.5 * delta)

print(f"At e = delta = {delta}:")
print(f"Quadratic branch (0.5 * e^2):        {quad_branch:.2f}")
print(f"Linear branch    (delta*(|e|-0.5d)): {lin_branch:.2f}")

# 3. Plot the three penalty curves
plt.figure(figsize=(8, 5))
plt.plot(e, half_mse_curve, label=r"$\frac{1}{2}e^2$ (Half MSE)", linestyle="--", linewidth=2)
plt.plot(e, mae_curve, label=r"$|e|$ (MAE)", linestyle=":", linewidth=2)
plt.plot(e, huber_curve, label=r"Huber ($\delta=1$)", linewidth=2.5)

# Mark the switch points at e = -1 and e = 1
plt.scatter([-delta, delta], [0.5, 0.5], color="red", zorder=5, label=r"Switch points ($e = \pm 1$)")

plt.title("Per-Error Penalty Curves: Half-MSE vs. MAE vs. Huber")
plt.xlabel("Error (e)")
plt.ylabel("Loss / Penalty")
plt.ylim(0, 8)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("step194_medium.png", dpi=150)
plt.show()