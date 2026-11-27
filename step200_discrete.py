from scipy.stats import bernoulli, binom, poisson, geom

# Binomial: 5 people walk past, each buys with p = 0.3
# probability that exactly 2 buy
print(binom.pmf(2, n=5, p=0.3))

# Poisson: on average 4 customers every 10 minutes
# probability that exactly 2 come in the next 10 minutes
print(poisson.pmf(2, mu=4))

# Geometric: each person buys with p = 0.3
# probability that the first buyer is the 3rd person
print(geom.pmf(3, p=0.3))

# cdf = cumulative: P(X <= k)
# probability that AT MOST 2 customers come in the next 10 minutes
print(poisson.cdf(2, mu=4))

# Verify: the Poisson pmf over 0..29 should add up to (almost exactly) 1
total = sum(poisson.pmf(k, mu=4) for k in range(30))
print(total)

import numpy as np

# Simulate 10,000 ten-minute windows at the tea kadai
counts = poisson.rvs(mu=4, size=10000, random_state=42)

# fraction of windows with exactly 2 customers
print(np.mean(counts == 2))

# average customers per window
print(counts.mean())