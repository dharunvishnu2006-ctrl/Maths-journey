# Checkpoint 1: Linear Algebra (Memory Recap)

### 1. Vectors & Dot Product
* **Idea:** Measures alignment and directional projection.
* **Formula:** $\mathbf{u} \cdot \mathbf{v} = \sum u_i v_i$
* **Hand:** $[2, 3] \cdot [4, 1] = 8 + 3 = 11$
* **NumPy:** `u @ v`

### 2. Matrix Multiplication
* **Idea:** Composes transformations via row-by-column dot products.
* **Rule:** $(m \times k) \times (k \times n) \to (m \times n)$
* **Hand:** $\begin{bmatrix}1 & 2\\3 & 4\end{bmatrix} \begin{bmatrix}5\\6\end{bmatrix} = \begin{bmatrix}5+12\\15+24\end{bmatrix} = \begin{bmatrix}17\\39\end{bmatrix}$
* **NumPy:** `A @ B`

### 3. Transpose & Inverse
* **Idea:** Transpose flips over diagonal; inverse reverses the transformation.
* **Rule:** $(A^T)_{ij} = A_{ji}$; $A A^{-1} = I$; $2\times 2$ inv: $\frac{1}{ad-bc}\begin{bmatrix}d & -b\\-c & a\end{bmatrix}$
* **Hand:** $A = \begin{bmatrix}4 & 7\\2 & 6\end{bmatrix} \implies A^T = \begin{bmatrix}4 & 2\\7 & 6\end{bmatrix}$; $\det = 10 \implies A^{-1} = \begin{bmatrix}0.6 & -0.7\\-0.2 & 0.4\end{bmatrix}$
* **NumPy:** `A.T`, `np.linalg.inv(A)`

### 4. Determinant
* **Idea:** Area/volume scaling factor; 0 means dimension collapse (singular).
* **Formula:** $\det\begin{pmatrix}a & b\\c & d\end{pmatrix} = ad - bc$
* **Hand:** $\det\begin{pmatrix}3 & 1\\2 & 5\end{pmatrix} = 15 - 2 = 13$
* **NumPy:** `np.linalg.det(A)`

### 5. Eigenvalues & Eigenvectors
* **Idea:** Axes where the transformation only stretches/shrinks, not rotates.
* **Formula:** $A\mathbf{v} = \lambda\mathbf{v}$, $\det(A - \lambda I) = 0$
* **Hand:** $\begin{bmatrix}2 & 0\\0 & 5\end{bmatrix}\begin{bmatrix}0\\1\end{bmatrix} = 5\begin{bmatrix}0\\1\end{bmatrix} \implies \lambda = 5, \mathbf{v} = [0, 1]^T$
* **NumPy:** `np.linalg.eigh(A)` (symmetric), `np.linalg.eig(A)`

### 6. The PCA Steps
* **Idea:** Projects data onto orthogonal axes of maximum variance.
* **Steps:** 
  1. $X_c = X - \boldsymbol{\mu}$
  2. $\Sigma = \frac{1}{n-1} X_c^T X_c$
  3. $\Sigma \mathbf{w}_i = \lambda_i \mathbf{w}_i$ (sorted descending)
  4. $W = [\mathbf{w}_1, \dots, \mathbf{w}_k]$ ($d \times k$)
  5. $Z = X_c W$ ($n \times k$)
* **Hand:** $X = \begin{bmatrix}1 & 3\\3 & 5\end{bmatrix} \implies \boldsymbol{\mu}=[2, 4], X_c = \begin{bmatrix}-1 & -1\\1 & 1\end{bmatrix} \implies \mathbf{w}_1 = [0.7071, 0.7071]^T \implies z_1 = -1.414$
* **NumPy:**
  ```python
  Xc = X - np.mean(X, axis=0)
  cov = (Xc.T @ Xc) / (len(X) - 1)
  vals, vecs = np.linalg.eigh(cov)
  W = vecs[:, np.argsort(vals)[::-1][:k]]
  Z = Xc @ W