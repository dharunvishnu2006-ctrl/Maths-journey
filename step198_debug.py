def bayes(prior, p_e_given_h, p_e_given_not_h):
    p_e = (p_e_given_h * prior +
           p_e_given_not_h * (1 - prior))
    print("p_e =", p_e)
    return p_e_given_h * prior / p_e

print("P(D|+) =", round(bayes(0.01, 0.99, 0.05), 4))
