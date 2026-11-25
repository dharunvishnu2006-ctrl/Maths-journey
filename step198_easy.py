import numpy as np

def cond_from_counts(n_a_and_b, n_b):
    if n_b == 0:
        raise ValueError("n_b must not be zero")
    return n_a_and_b / n_b

total = 2000
night = 300
fraud = 24
fraud_night = 18

p_fraud_given_night = cond_from_counts(fraud_night, night)
p_night_given_fraud = cond_from_counts(fraud_night, fraud)
p_fraud = fraud / total

print("P(fraud | night) =", p_fraud_given_night)
print("P(night | fraud) =", p_night_given_fraud)
print("P(fraud) =", p_fraud)
print("Independent?", np.isclose(p_fraud_given_night, p_fraud))