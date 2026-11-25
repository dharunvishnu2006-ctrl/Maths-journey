# Step 198: Conditional Probability
import numpy as np

# --- 1. Conditional probability (class of 40) ---
total = 40
p_c = 25 / total          # P(cricket)
p_h = 15 / total          # P(chess)
p_c_and_h = 8 / total     # P(cricket AND chess)

def conditional(p_a_and_b, p_b):
    """P(A|B) = P(A and B) / P(B)"""
    if p_b <= 0:
        raise ValueError("P(B) must be > 0")
    return p_a_and_b / p_b

p_c_given_h = conditional(p_c_and_h, p_h)
print("P(C|H) =", round(p_c_given_h, 4))
print("P(H|C) =", round(conditional(p_c_and_h, p_c), 4))

# --- 2. Independence check ---
print("Independent?", np.isclose(p_c_given_h, p_c))

# --- 3. Bayes theorem (medical test) ---
def bayes(prior, p_pos_given_d, p_pos_given_not_d):
    """P(D|+) = P(+|D) * P(D) / P(+)"""
    # P(+) by the law of total probability
    p_pos = p_pos_given_d * prior + p_pos_given_not_d * (1 - prior)
    return p_pos_given_d * prior / p_pos

print("P(D|+) =", round(bayes(0.01, 0.99, 0.05), 4))

# --- 4. Base-rate demo: SAME test, different priors ---
for prior in [0.001, 0.01, 0.1, 0.5]:
    print("prior =", prior, "-> P(D|+) =", round(bayes(prior, 0.99, 0.05), 4))

# --- 5. Verify Bayes with a simulation ---
rng = np.random.default_rng(42)
n = 1_000_000
has_d = rng.random(n) < 0.01                  # 1% have the disease
p_pos = np.where(has_d, 0.99, 0.05)           # each person's chance of a positive
positive = rng.random(n) < p_pos              # who actually tests positive

sim = np.mean(has_d[positive])                # among positives, share who are sick
print("Simulated P(D|+) =", round(sim, 4))
print("Share of positives that are FALSE =", round(1 - sim, 4))    