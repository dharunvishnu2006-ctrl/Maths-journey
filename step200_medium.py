from scipy.stats import poisson

lam = 3 * (20 / 60)

print(1 - poisson.cdf(1, mu=lam))