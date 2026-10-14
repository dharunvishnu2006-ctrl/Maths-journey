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