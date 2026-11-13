import numpy as np
import matplotlib.pyplot as plt

def loss_fn(w):
    return 0.5 * (w[0]**2 + 50.0 * w[1]**2)

def grad_fn(w):
    return np.array([w[0], 50.0 * w[1]], dtype=float)

def run_optimizer(w_init, alpha=0.02, beta=0.9, mode="gd", max_steps=2000, tol=1e-6):
    w = np.array(w_init, dtype=float)
    v = np.zeros_like(w)
    losses, target_step, diverged = [], None, False

    for step in range(max_steps + 1):
        curr_loss = loss_fn(w)
        if not np.isfinite(curr_loss) or curr_loss > 1e12:
            diverged = True
            break
        losses.append(max(curr_loss, 1e-16))
        if target_step is None and curr_loss < tol:
            target_step = step
        if step == max_steps:
            break

        if mode == "gd":
            w = w - alpha * grad_fn(w)
        elif mode == "momentum":
            v = beta * v - alpha * grad_fn(w)
            w = w + v
        elif mode == "nesterov":
            v = beta * v - alpha * grad_fn(w + beta * v)
            w = w + v
    return target_step, diverged, losses

alpha = 0.02
configs = [
    ("Plain GD", "gd", 0.0),
    ("Momentum (β=0.9)", "momentum", 0.9),
    ("Nesterov (β=0.9)", "nesterov", 0.9),
]
results = []

plt.figure(figsize=(8, 4.5))
for label, mode, beta in configs:
    steps, div, losses = run_optimizer([10.0, 1.0], alpha=alpha, beta=beta, mode=mode)
    status = "DIVERGED" if div else f"{steps} steps"
    print(f"{label:<18}: {status}")
    plt.semilogy(losses, label=f"{label} ({status})", linewidth=2)
    if not div and steps is not None:
        results.append((steps, label))

plt.axhline(1e-6, color="red", linestyle="--", label="Target (1e-6)")
plt.xlim(0, 500)
plt.ylim(1e-12, 1e3)
plt.xlabel("Step")
plt.ylabel("Loss (log scale)")
plt.title(f"Narrow Valley Optimization (α = {alpha})")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("step192_app_chart.png", dpi=150)

best_steps, best_label = min(results)
print(f"\nRecommendation (α={alpha}): {best_label} — hits target in {best_steps} steps.")