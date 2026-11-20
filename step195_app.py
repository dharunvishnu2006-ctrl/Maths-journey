import numpy as np

# 1. Build the imbalanced UPI dataset: 990 normal rows (y=0, p=0.05) and 10 fraud rows (y=1, p=0.30).
y = np.array([0] * 990 + [1] * 10)
p = np.array([0.05] * 990 + [0.30] * 10)


# 2. Define focal_loss(y, p, gamma, eps=1e-15): clip p, compute p_t, apply -(1 - p_t)^gamma * log(p_t), and return per-row losses.
def focal_loss(y, p, gamma=2.0, eps=1e-15):
  p = np.clip(p, eps, 1.0 - eps)
  p_t = np.where(y == 1, p, 1.0 - p)
  return -((1.0 - p_t) ** gamma) * np.log(p_t)


def bce_rows(y, p, eps=1e-15):
  p = np.clip(p, eps, 1.0 - eps)
  return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


# 3. Verify the gamma = 0 edge case: assert that focal_loss(y, p, gamma=0) matches standard BCE within floating-point tolerance.
focal_0_mean = np.mean(focal_loss(y, p, gamma=0))
bce_mean = np.mean(bce_rows(y, p))
assert np.isclose(
    focal_0_mean, bce_mean
), f"Mismatch: {focal_0_mean} vs {bce_mean}"
print(
    f"Check gamma=0 vs BCE: focal={focal_0_mean:.6f}, bce={bce_mean:.6f}"
    " (Match: True)\n"
)

# 4. Loop over gamma in [0, 1, 2, 5]: compute total loss, fraud-only loss (guarding against 0/0), and print the fraud share percentage.
fraud_mask = y == 1
for gamma in [0, 1, 2, 5]:
  losses = focal_loss(y, p, gamma=gamma)
  mean_loss = np.mean(losses)
  total_loss = np.sum(losses)
  fraud_loss = np.sum(losses[fraud_mask])
  fraud_share = (fraud_loss / total_loss * 100.0) if total_loss > 0 else 0.0
  print(
      f"gamma={gamma} | mean_loss={mean_loss:.6f} | fraud_share={fraud_share:.2f}%"
  )

# 5. Print the recommended gamma value and why focal loss solves class imbalance in fraud detection.
print("\nRecommendation: gamma = 2")
print(
    "Why: At gamma=0, 990 easy normal transactions generate ~80% of the loss"
    " and drown out fraud."
)
print(
    "At gamma=2, easy normal predictions are suppressed by 99.75% while hard"
    " fraud cases retain"
)
print(
    "nearly all the gradient signal (~98% of total loss), without overly"
    " starving borderline cases like gamma=5."
)