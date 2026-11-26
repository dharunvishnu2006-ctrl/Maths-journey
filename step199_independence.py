import numpy as np

def is_independent(p_a, p_b, p_ab, tol=0.01):
    """Product test: P(A and B) == P(A) * P(B), within a tolerance."""
    return abs(p_ab - p_a * p_b) <= tol

# --- 1. Pairwise vs mutual: two coins, simulated ---
rng = np.random.default_rng(42)
n = 100_000
coin1 = rng.random(n) < 0.5      # True = Heads
coin2 = rng.random(n) < 0.5

A = coin1                 # first is Heads
B = coin2                 # second is Heads
C = coin1 == coin2        # both show the same face

pA, pB, pC = A.mean(), B.mean(), C.mean()
print("A,B independent?", is_independent(pA, pB, (A & B).mean()))
print("A,C independent?", is_independent(pA, pC, (A & C).mean()))
print("B,C independent?", is_independent(pB, pC, (B & C).mean()))

p_abc = (A & B & C).mean()
print("P(A&B&C) =", round(p_abc, 4), " vs  P(A)P(B)P(C) =", round(pA * pB * pC, 4))

# --- 2. Conditional independence: the exam ---
easy = rng.random(n) < 0.5                  # True = Easy paper
p_pass = np.where(easy, 0.9, 0.3)           # each exam's pass chance
ravi = rng.random(n) < p_pass
priya = rng.random(n) < p_pass              # separate random draw = no copying

def indep_in(mask, X, Y):
    """Product test inside the 'room' where mask is True."""
    x, y = X[mask], Y[mask]
    return is_independent(x.mean(), y.mean(), (x & y).mean())

everyone = np.ones(n, dtype=bool)           # a mask that keeps every exam
print("Overall independent?   ", indep_in(everyone, ravi, priya))
print("Independent given Easy?", indep_in(easy, ravi, priya))
print("Independent given Hard?", indep_in(~easy, ravi, priya))