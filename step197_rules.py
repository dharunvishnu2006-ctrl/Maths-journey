# Step 197: Probability Rules
import numpy as np

# --- Rule 1: Addition rule (class of 40) ---
total = 40
p_c = 25 / total      # P(cricket)
p_h = 15 / total      # P(chess)
p_ch = 8 / total      # P(cricket AND chess)

p_c_or_h = p_c + p_h - p_ch
print("P(C or H) =", p_c_or_h)

# --- Rule 2: Complement ---
p_neither = 1 - p_c_or_h
print("P(neither) =", round(p_neither, 4))

# --- Verify the addition rule with a simulation ---
rng = np.random.default_rng(42)
n = 100_000
students = rng.integers(0, 40, size=n)   # pick a random student (0..39), n times

# students 0-24 play cricket; students 0-7 and 25-31 play chess
cricket = students < 25
chess = (students < 8) | ((students >= 25) & (students < 32))

sim_or = np.mean(cricket | chess)
print("Simulated P(C or H) =", round(sim_or, 4))
print("Formula   P(C or H) =", round(p_c_or_h, 4))

# --- Rule 3: Multiplication rule (laddoo box) ---
p_first = 4 / 10                 # P(1st is laddoo)
p_second_given_first = 3 / 9     # P(2nd is laddoo | 1st was laddoo)
p_both = p_first * p_second_given_first
print("P(both laddoos) =", round(p_both, 4))

# --- Rule 4: Inclusion-exclusion for 3 events (class of 40) ---
c, h, k = 25, 15, 10
ch, ck, hk = 8, 5, 3
chk = 2
at_least_one = (c + h + k - ch - ck - hk + chk) / total
print("P(at least one sport) =", round(at_least_one, 4))
print("P(no sport) =", round(1 - at_least_one, 4))

# --- Rule 5: Law of total probability (delivery apps) ---
p_app = np.array([0.6, 0.4])          # P(X), P(Y)  - the routes
p_late_given = np.array([0.1, 0.2])   # P(Late|X), P(Late|Y)

print("Routes sum to 1?", np.isclose(p_app.sum(), 1.0))
p_late = np.sum(p_late_given * p_app)
print("P(Late) =", round(p_late, 4))
print("P(On time) =", round(1 - p_late, 4))