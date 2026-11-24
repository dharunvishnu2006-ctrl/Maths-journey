def prob_or(p_a, p_b, p_ab):
    return p_a + p_b - p_ab

p_fraud = 0.02
p_location = 0.05
p_both = 0.01

p_either = prob_or(p_fraud, p_location, p_both)
p_neither = 1 - p_either

print("Fraud or unusual location =", p_either)
print("Neither =", p_neither)