def combine_evidence(prior, signals):

    if not 0 <= prior <= 1:
        raise ValueError("Invalid prior")

    for p_fraud, p_genuine in signals:

        if not (0 <= p_fraud <= 1 and
                0 <= p_genuine <= 1):
            raise ValueError("Invalid signal probability")

        # Total probability
        p_signal = (p_fraud * prior +
                    p_genuine * (1 - prior))

        if p_signal == 0:
            raise ValueError("Evidence probability is zero")

        # Bayes update
        prior = (p_fraud * prior) / p_signal

        print("Updated probability:", round(prior, 4))

    return prior


# Step 2: Main program
prior = 0.001

signals = [
    (0.60, 0.05),  # Amount > ₹1,00,000
    (0.40, 0.02),  # Time = 3 am
    (0.70, 0.10)   # New device
]

final_probability = combine_evidence(prior, signals)

if final_probability > 0.5:
    print("Decision: BLOCK")
else:
    print("Decision: ALLOW")


# Step 3: Edge case 1 - Empty signals
print("\nTest 1: Empty signals")
print("Result:", combine_evidence(0.001, []))


# Step 4: Edge case 2 - Prior = 0
print("\nTest 2: Prior = 0")
print("Result:", combine_evidence(0, signals))