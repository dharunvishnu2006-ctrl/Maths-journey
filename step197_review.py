import numpy as np

# Function 1: Complement rule
def at_least_one_complement(probs):

    # Check empty input
    if len(probs) == 0:
        raise ValueError("Empty probability list")

    # Check valid probabilities
    if any(p < 0 or p > 1 for p in probs):
        raise ValueError("Invalid probability")

    # Calculate probability that none fail
    product = 1
    for p in probs:
        product *= (1 - p)

    # Apply complement rule
    return 1 - product


# Function 2: Inclusion-exclusion for 3 events
def at_least_one_incl_excl(p1, p2, p3):

    return (
        p1 + p2 + p3
        - p1*p2
        - p1*p3
        - p2*p3
        + p1*p2*p3
    )


# Main program
probs = [0.1, 0.2, 0.3]

result1 = at_least_one_complement(probs)
result2 = at_least_one_incl_excl(*probs)

print("Complement =", round(result1, 4))
print("Inclusion-exclusion =", round(result2, 4))
print("Agree =", np.isclose(result1, result2))

# PREDICT:
# Test 1: Empty probability list
# Test 2: Invalid probability

try:
    at_least_one_complement([])
except ValueError as e:
    print("Caught:", e)

try:
    at_least_one_complement([0.5, 1.2])
except ValueError as e:
    print("Caught:", e)
    