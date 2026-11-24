def prob_and(p_a, p_b_given_a):
    return p_a * p_b_given_a

p_both = prob_and(3/12, 2/11)
p_at_least_one = 1 - (9/12 * 8/11)
p_with_replacement = prob_and(3/12, 3/12)

print("Both faulty =", round(p_both, 4))
print("At least one faulty =", round(p_at_least_one, 4))
print("Both faulty with replacement =", round(p_with_replacement, 4))