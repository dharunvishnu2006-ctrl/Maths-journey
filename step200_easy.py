from scipy.stats import binom

# P(no wicket)
print(binom.pmf(0, n=6, p=0.1))

# P(at least one wicket) using complement
print(1 - binom.pmf(0, n=6, p=0.1))

# P(at least one wicket) using CDF
print(1 - binom.cdf(0, n=6, p=0.1))