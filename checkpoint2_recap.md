1. Derivative Rules
Power: d(xⁿ)/dx = n × xⁿ⁻¹
Product: f'g + fg'
Quotient: (f'g − fg') / g²
eˣ: eˣ
ln(x): 1/x
Sigmoid: σ(x) × (1 − σ(x))
2. Chain Rule
Formula: Outer derivative × Inner derivative.
Example: y = (3x + 2)³
y' = 9(3x + 2)²
At x = 1: 225
3. Partial Derivatives and Gradient
Partial derivative: Differentiate one variable while keeping others constant.
Gradient: Vector of all partial derivatives.
Example: f(x,y) = 3x² + 2y
Gradient = [6x, 2]
At (1,4): [6, 2]
4. Gradient Descent
Update: w = w − α × gradient
α (learning rate): Controls step size.
5. Momentum and Nesterov

Momentum:

v = β × previous_v − α × gradient
w = previous_w + v
β: Controls previous velocity retention.

Nesterov:

Calculates gradient at the look-ahead position.
v = β × previous_v − α × gradient( w + β × previous_v )
w = previous_w + v
6. Adam Optimiser

First Moment:

m = β1 × previous_m + (1 − β1) × gradient

Second Moment:

v = β2 × previous_v + (1 − β2) × gradient²

Bias Correction:

m_hat = m / (1 − β1^t)
v_hat = v / (1 − β2^t)

Weight Update:

w = previous_w − α × m_hat / (√v_hat + ε)

Default Values:

α = 0.001
β1 = 0.9
β2 = 0.999
ε = 1e−8
7. Regression Losses

MSE:

Average of squared errors.
MSE = Σ(actual − predicted)² / n

MAE:

Average of absolute errors.
MAE = Σ|actual − predicted| / n

Huber Loss:

If |error| ≤ δ: Loss = 0.5 × error²
If |error| > δ: Loss = δ × (|error| − 0.5 × δ)
Final Huber = Average of individual losses.

R²:

R² = 1 − (SS_res / SS_tot)
SS_res = Σ(actual − predicted)²
SS_tot = Σ(actual − actual_mean)²
8. Classification Losses

BCE — Binary Cross-Entropy:

BCE = −[y × ln(p) + (1 − y) × ln(1 − p)]

CCE — Categorical Cross-Entropy:

CCE = −Σ(yₖ × ln(pₖ))

Hinge Loss:

Loss = max(0, 1 − y × score)
y belongs to {-1, +1}.

Focal Loss:

Loss = −(1 − p_t)^γ × ln(p_t)
Focuses more on difficult examples.