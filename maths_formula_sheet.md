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