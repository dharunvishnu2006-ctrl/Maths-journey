def fraud_posterior(prior, p_alert_given_fraud,
                    p_alert_given_genuine):
    p_alert = (p_alert_given_fraud * prior +
               p_alert_given_genuine * (1 - prior))
    return (p_alert_given_fraud * prior) / p_alert

first = fraud_posterior(0.002, 0.95, 0.02)

second = fraud_posterior(first, 0.95, 0.02)

print("P(fraud | alert) =", round(first, 4))
print("P(fraud | both alerts) =", round(second, 4))