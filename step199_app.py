import numpy as np

# 1. Given counts
spam = 200
normal = 800
total = 1000

# 2. Conditional independence helper
def conditional_independence(both, offer, free, group):
    if group == 0:
        raise ValueError("Group cannot be zero")

    p_both = both / group
    p_product = (offer / group) * (free / group)

    return np.isclose(p_both, p_product)


# 3. Test conditional independence
print("Given spam:",
      conditional_independence(90, 120, 100, spam))

print("Given normal:",
      conditional_independence(4, 40, 80, normal))


# 4. Naive Bayes calculation
nb_spam = (120/200) * (100/200) * (spam/total)
nb_normal = (40/800) * (80/800) * (normal/total)

nb_denominator = nb_spam + nb_normal
nb = nb_spam / nb_denominator


# 5. True posterior from actual counts
true_posterior = 90 / (90 + 4)


# 6. Compare results
print("Naive Bayes:", round(nb, 4))
print("True posterior:", round(true_posterior, 4))

if nb < true_posterior:
    print("Naive Bayes is too LOW")
elif nb > true_posterior:
    print("Naive Bayes is too HIGH")
else:
    print("Both are equal")


# 7. Edge-case test
try:
    conditional_independence(0, 0, 0, 0)
except ValueError as error:
    print("Edge case handled:", error)


# 8. Reflection
# Time complexity: O(1)
# Weakest part: Conditional independence assumption
