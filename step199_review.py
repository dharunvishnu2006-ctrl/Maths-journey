import numpy as np

def check_three(A, B, C, tol=0.01):

    # Check edge cases
    if len(A) == 0:
        raise ValueError("Arrays cannot be empty")

    if len(A) != len(B) or len(B) != len(C):
        raise ValueError("Arrays must have equal lengths")

    # Individual probabilities
    pA = A.mean()
    pB = B.mean()
    pC = C.mean()

    # Pairwise tests
    ab = np.isclose((A & B).mean(), pA * pB, atol=tol)
    ac = np.isclose((A & C).mean(), pA * pC, atol=tol)
    bc = np.isclose((B & C).mean(), pB * pC, atol=tol)

    # Triple test
    abc = np.isclose(
        (A & B & C).mean(),
        pA * pB * pC,
        atol=tol
    )

    print("A,B:", ab)
    print("A,C:", ac)
    print("B,C:", bc)
    print("Triple:", abc)

    return ab and ac and bc and abc


# Simulation 1: Independent fair coins
rng = np.random.default_rng(42)
n = 100_000

A = rng.random(n) < 0.5
B = rng.random(n) < 0.5
C = rng.random(n) < 0.5

print("\nIndependent coins:")
print("Mutually independent:", check_three(A, B, C))


# Simulation 2: Sensor setup
rng = np.random.default_rng(42)

S1 = rng.random(n) < 0.5
S2 = rng.random(n) < 0.5
S3 = S1 != S2

print("\nSensor setup:")
print("Mutually independent:", check_three(S1, S2, S3))
