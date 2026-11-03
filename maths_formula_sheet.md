## Step 179: Determinant of a 2x2 Matrix & Invertibility

### Formula
$$\det(A) = |A| = ad - bc$$

### Symbols
- $A$: A square matrix $\begin{bmatrix} a & b \\ c & d \end{bmatrix}$ representing a linear transformation.
- $a, d$: Elements along the main diagonal.
- $b, c$: Elements along the anti-diagonal.
- $\det(A)$: The scalar factor by which the transformation scales area (signed: positive preserves orientation, negative flips it, zero collapses area to a line or point, making $A$ singular/non-invertible).

### Worked Example
$$A = \begin{bmatrix} 3 & 2 \\ 1 & 4 \end{bmatrix}$$

$$\det(A) = (3)(4) - (2)(1) = 12 - 2 = 10$$

Since $\det(A) = 10 \neq 0$, the matrix scales area by a factor of 10, preserves orientation, and is invertible.

## Step 180: Eigenvalue Equation & Principal Direction

### Formula
$$A v = \lambda v$$

### Symbols
- $A$: Square matrix (transformation or covariance matrix).
- $v$: Non-zero eigenvector (direction whose orientation remains unchanged under transformation).
- $\lambda$: Eigenvalue (scalar scale factor stretching or shrinking $v$).

### Worked Example
For $A = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$ and $v = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$:
$$A v = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix} \begin{bmatrix} 1 \\ 1 \end{bmatrix} = \begin{bmatrix} 3 \\ 3 \end{bmatrix} = 3 \begin{bmatrix} 1 \\ 1 \end{bmatrix}$$
Here, $\lambda = 3$ is the dominant eigenvalue and $v = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$ is its corresponding eigenvector.

## Step 181: PCA Foundations

### Formulas
1. $X_c = X - \mathbf{1}\boldsymbol{\mu}^T$
2. $C = \frac{1}{n-1}X_c^T X_c$
3. $r_i = \frac{\lambda_i}{\sum \lambda}$

### Symbols
* $X$: Data matrix ($n \times d$)
* $n, d$: Sample count, feature count
* $\boldsymbol{\mu}$: Column means ($d \times 1$)
* $X_c$: Centred matrix ($n \times d$)
* $C$: Sample covariance ($d \times d$)
* $\lambda_i$: Eigenvalue $i$ (sorted descending)
* $r_i$: Explained variance ratio

### Worked Example ($X = [[1, 3], [3, 1], [5, 7], [7, 5]]$)
1. $\boldsymbol{\mu} = [4, 4]$
2. $X_c = [[-3, -1], [-1, -3], [1, 3], [3, 1]]$
3. $C = \frac{1}{3}\begin{bmatrix} 20 & 12 \\ 12 & 20 \end{bmatrix}$
4. $\lambda = [32/3, 8/3] \approx [10.67, 2.67]$
5. $r = [0.80, 0.20]$

## Step 182: PCA Projection & Loadings

### Formulas
* $X_c = X - \boldsymbol{\mu}$
* $Z = X_c W$
* Loadings in manual $W$: columns of $W$ ($d \times k$).
* Loadings in sklearn: rows of `pca.components_`, so $W = \text{pca.components\_}[:k].T$.
* Sklearn projection: $Z = (X - \text{pca.mean\_}) @ \text{pca.components\_}[:k].T$

### Symbols
* $X$: Raw input data $(n \times d)$
* $\boldsymbol{\mu}$: Feature mean vector $(d,)$
* $X_c$: Mean-centered data $(n \times d)$
* $W$: Weight/loading matrix $(d \times k)$
* $Z$: Projected score matrix $(n \times k)$
* `pca.components_`: Principal axes as rows $(k \times d)$
* $k$: Number of retained components

### Example (Tailor Size Axis)
* Point: $[42, 34]$, Mean: $\boldsymbol{\mu} = [40, 32]$
* $X_c = [42 - 40, 34 - 32] = [2, 2]$
* Weight $\mathbf{w} = [0.7071, 0.7071]^T$
* Score $z = 2(0.7071) + 2(0.7071) = \mathbf{2.8284}$

## Step 183: SVD Decomposition & Truncated SVD

### Formulas

* $A = U \Sigma V^T$
* Rank-$k$ reconstruction: $A_k = U_{:, :k} \, \Sigma_{:k, :k} \, V_{:k, :}^T = \sum_{i=1}^k \sigma_i \mathbf{u}_i \mathbf{v}_i^T$
* Reconstruction error (Frobenius): $\Vert{}A - A_k\Vert{}_F = \sqrt{\sigma_{k+1}^2 + \dots + \sigma_{\min(m, n)}^2}$
* Share kept: $\frac{\sum_{i=1}^k \sigma_i^2}{\sum_{i=1}^{\min(m, n)} \sigma_i^2} \times 100\%$

### Symbols

* $A$: Input matrix $(m \times n)$
* $U$: Left singular vectors / output directions $(m \times m)$
* $\Sigma$: Diagonal rectangular matrix of singular values $(m \times n)$
* $V^T$: Right singular vectors / input directions $(n \times n)$
* $\sigma_i$: $i$-th singular value ($\sqrt{\lambda_i(A^T A)}$), non-negative and sorted descending
* $k$: Number of top singular values retained
* $A_k$: Best rank-$k$ approximation of $A$ (Eckart–Young theorem)

### Example ($2 \times 2$ Matrix)

* Matrix: $A = \begin{bmatrix} 0 & 3 \\ 0 & 0 \end{bmatrix}$
* $A^T A = \begin{bmatrix} 0 & 0 \\ 0 & 9 \end{bmatrix} \implies \sigma_1 = 3, \; \sigma_2 = 0$
* Vectors: $\mathbf{v}_1 = \begin{bmatrix} 0 \\ 1 \end{bmatrix}, \quad \mathbf{u}_1 = \frac{A \mathbf{v}_1}{\sigma_1} = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$
* Rank-1 approximation ($k = 1$):

$$A_1 = 3 \begin{bmatrix} 1 \\ 0 \end{bmatrix} \begin{bmatrix} 0 & 1 \end{bmatrix} = \begin{bmatrix} 0 & 3 \\ 0 & 0 \end{bmatrix} = A$$


* Share kept: $\mathbf{100\%}$, Error: $\mathbf{0}$

## Step 184: Linear Systems, Normal Equations & Condition Number

### Formulas

* Exact system: `A x = b`  (solution: `x = A^-1 b` when `A` is square and invertible)
* Normal equation (Least Squares): `A^T A x = A^T b`
* Condition number: `cond(A) = sigma_max / sigma_min`

---

### Symbols

* `A`: Coefficient matrix of shape `(m, n)` representing equations and feature weights
* `x`: Unknown vector of shape `(n,)` to solve for
* `b`: Target constants or observed totals vector of shape `(m,)`
* `A^T`: Transpose of matrix `A`, shape `(n, m)`
* `A^T A`: Normal equation matrix, always square `(n, n)`
* `sigma_max`: Largest singular value of `A` (maximum stretch)
* `sigma_min`: Smallest singular value of `A` (minimum stretch)
* `cond(A)`: Ratio measuring sensitivity/error magnification of the system

---

### Hand-Worked Example

System:

1. `x + y = 5`
2. `x + 2y = 8`

Matrix form `A x = b`:
`A = [[1, 1], [1, 2]]`, `b = [5, 8]`

1. **Determinant of `A`:**
`det(A) = (1)(2) - (1)(1) = 2 - 1 = 1`
2. **Inverse `A^-1`:**
`A^-1 = (1 / 1) * [[2, -1], [-1, 1]] = [[2, -1], [-1, 1]]`
3. **Solve for `x`:**
`x = A^-1 b = [[2, -1], [-1, 1]] @ [5, 8]`
`x_1 = 2(5) - 1(8) = 10 - 8 = 2`
`x_2 = -1(5) + 1(8) = -5 + 8 = 3`

**Solution:**
`x = 2`, `y = 3`

**Verification:**
`2 + 3 = 5` and `2 + 2(3) = 8`

## Step 185: Derivatives from First Principles & the Power Rule

### Formulas & Definitions

1. **Derivative (Limit Definition):**
   `f'(x) = limit as h -> 0 of (f(x + h) - f(x)) / h`
   * `f(x + h) - f(x)`: The change in output value (vertical rise).
   * `h`: The tiny step size along the input axis (horizontal run).
   * `(f(x + h) - f(x)) / h`: Average rate of change across the small gap `h`.
   * `limit as h -> 0`: Shrinks the gap to zero, turning the secant slope into the exact instantaneous slope at point `x`.

2. **The Power Rule:**
   `d/dx (x^n) = n * x^(n - 1)`
   * Bring the current exponent `n` to the front as a multiplier, then decrease the power by 1.

3. **Constant Rule:**
   `d/dx (c) = 0`
   * A flat constant has zero change, so its slope is always 0.

---

### Hand-Worked Example

Function: `f(x) = 2x^2 + 5x` evaluated at `x = 3`

1. **Using the Limit Definition:**
   * `f(3) = 2(3)^2 + 5(3) = 2(9) + 15 = 33`
   * `f(3 + h) = 2(3 + h)^2 + 5(3 + h)`
     `= 2(9 + 6h + h^2) + 15 + 5h`
     `= 18 + 12h + 2h^2 + 15 + 5h`
     `= 33 + 17h + 2h^2`
   * `f(3 + h) - f(3) = (33 + 17h + 2h^2) - 33 = 17h + 2h^2`
   * Divide by `h`:
     `(17h + 2h^2) / h = 17 + 2h`
   * Take `limit as h -> 0`:
     `17 + 2(0) = 17`

2. **Verification via Power Rule:**
   * `f'(x) = 2 * (2x^(2-1)) + 5 * (1x^(1-1))`
   * `f'(x) = 4x + 5`
   * At `x = 3`: `4(3) + 5 = 12 + 5 = 17`

   ## Step 186: Product Rule, Quotient Rule & Chain Rule

### Formulas

1. **Product Rule:**
   `(f * g)' = f' * g + f * g'`
   * **When to use:** Use when two separate variable expressions are multiplied together.

2. **Quotient Rule:**
   `(f / g)' = (f' * g - f * g') / g^2`
   * **When to use:** Use when one variable expression is divided by another (fraction form).

3. **Chain Rule:**
   `[f(g(x))]' = f'(g(x)) * g'(x)`  (outer' * inner')
   * **When to use:** Use when an expression is nested inside another function (composite function, like an expression raised to a power).

---

### Hand-Worked Example (Chain Rule)

Function: `y = (3x + 2)^3` evaluated at `x = 1`

1. **Break into outer and inner:**
   * Outer function: `f(u) = u^3` where `u = 3x + 2`
   * Inner function: `u(x) = 3x + 2`

2. **Differentiate each piece:**
   * Derivative of outer with respect to `u`: `f'(u) = 3 * u^2 = 3 * (3x + 2)^2`
   * Derivative of inner with respect to `x`: `u'(x) = 3`

3. **Multiply outer' by inner':**
   * `y' = f'(u) * u'(x)`
   * `y' = 3 * (3x + 2)^2 * 3 = 9 * (3x + 2)^2`

4. **Evaluate at `x = 1`:**
   * `y'(1) = 9 * (3(1) + 2)^2`
   * `y'(1) = 9 * (5)^2 = 9 * 25 = 225`

## Step 187: Partial Derivatives and Gradients

### Partial Derivative
- ∂f/∂x measures how fast the output f changes when only x moves.
- All other variables (like y and z) are treated as plain constant numbers.

### Gradient (∇f)
- ∇f = [∂f/∂x, ∂f/∂y, ...]
- It bundles all individual partial derivatives into a single direction vector.
- It always points in the direction of steepest uphill increase.

### Symbols
- ∂ (del / partial): means rate of change with respect to only one input while freezing the others.
- ∇ (nabla / grad): turns a function into a vector of all its partial derivatives.
- h (step size): a tiny number (like 1e-5) used to nudge an input slightly for numerical approximation.

### Worked Example
- Function: f(x, y) = 3*x^2 + 2*y
- Point: (1, 4)

1. Find partial derivatives:
   - ∂f/∂x = 6*x + 0 = 6*x
   - ∂f/∂y = 0 + 2 = 2

2. Gradient formula:
   - ∇f(x, y) = [6*x, 2]

3. Evaluate at (1, 4):
   - ∂f/∂x at 1 = 6*(1) = 6
   - ∂f/∂y at 4 = 2
   - ∇f(1, 4) = [6, 2]   

## Step 188: Backpropagation and Multi-Path Accumulation

### The Chain Rule
dy/dx = (dy/du) * (du/dx)
Why rates multiply: If y changes 3 times as fast as u, and u changes 2 times as fast as x, then a 1-unit nudge in x shifts u by 2, which shifts y by 2 * 3 = 6. Scaling factors multiply along a chain.

### The Backward Pass
- Starts at: The final loss node L with dL/dL = 1.0 (the rate of change of any quantity with respect to itself is always 1).
- Direction: Walks backward against the direction of the computational graph arrows, from the scalar loss back to the inputs and weights.
- At each arrow: Incoming upstream gradient * local rate of that specific operation = downstream gradient passed to the preceding node.

### Accumulation
- When a variable splits and feeds into multiple paths, its total derivative is the sum of the contributions from each branch: total dL/dw = path_1 + path_2.
- The += bug: Assigning with `=` inside code instead of `+=` silently overwrites the gradient accumulated from earlier paths, discarding entire branches of the network.

### Symbols
- L: Final scalar loss value to minimize.
- w: Model weight or parameter receiving gradients.
- x: Input feature or data value.
- z: Intermediate pre-activation or linear combination node.
- a: Activated output node (e.g., sigmoid(z)).
- dL/d(node): Upstream gradient measuring how the total loss changes with respect to that node.
- d(out)/d(in): Local derivative of an operation holding all other inputs fixed.

### Tiny Worked Example
Function: L = (w * x) + w, at w = 2, x = 3
- Forward pass:
  - m = w * x = 2 * 3 = 6
  - L = m + w = 6 + 2 = 8
- Backward pass:
  - Seed: dL/dL = 1
  - Local rate for m: dL/dm = 1
  - Local rate for direct w: dL/dw (path 2) = 1
  - Path 1 (via m): dL/dm * dm/dw = 1 * x = 1 * 3 = 3
  - Path 2 (direct): dL/dL * 1 = 1 * 1 = 1
  - Total dL/dw = Path 1 + Path 2 = 3 + 1 = 4
  - Check: L = w*(x + 1) = w*(4) -> dL/dw = 4   

## Step 189: Gradient Descent Update Rule

- Update Rule: w = w - alpha * (dL/dw)
  Why minus: The derivative points uphill (steepest increase). Subtracting it forces the step downhill toward the lowest loss.

- Symbols:
  - w: The model weight/parameter being tuned.
  - alpha: The learning rate, controlling how large of a step to take.
  - dL/dw: The gradient (slope) of the loss with respect to w.
  - L: The loss function measuring how far predictions are from true values.

- Learning Rate (alpha) and the (1 - 2*alpha) Step Multiplier:
  - Good (0 < alpha < 1.0, optimal near 0.5): Shrinks the error distance by a fraction each step.
  - Too small (e.g., 0.01): The multiplier (1 - 2*alpha) is close to 1; steps are tiny and convergence is very slow.
  - Too big (alpha > 1.0): |1 - 2*alpha| > 1; steps overshoot the valley and explode to infinity.

- Local vs Global Minimum:
  - A local minimum is the lowest point in an immediate neighborhood; a global minimum is the absolute lowest point across the entire function.

- Hand-Worked Example:
  - Loss: L(w) = (w + 2)^2, so dL/dw = 2w + 4. Start at w0 = 0 with alpha = 0.25.
  - Step 1: dL/dw = 2(0) + 4 = 4.
    w1 = 0 - 0.25 * 4 = -1.0.
  - Step 2: dL/dw = 2(-1.0) + 4 = 2.
    w2 = -1.0 - 0.25 * 2 = -1.5. (Approaching the minimum at w* = -2.0).
