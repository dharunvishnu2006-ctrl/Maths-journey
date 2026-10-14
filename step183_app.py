import numpy as np

# 1. Create task image (20, 20)
x = np.arange(20)
img = np.add.outer(x, x) * 5.0
img[5:15, 5:15] += 100

# Print requested check values
print("img[0, :3]:", img[0, :3], "img[10, 10]:", img[10, 10])

def svd_compress(matrix, threshold=0.90):
    # 2. Decompose
    U, S, Vt = np.linalg.svd(matrix)

    # 3. Zero check
    total_energy = np.sum(S**2)
    if total_energy == 0:
        return 1, 100.0, 0.0, 0.0, True

    # 4. Cumulative energy share
    cum_energy = np.cumsum(S**2) / total_energy

    # 5. Smallest k reaching threshold
    k = int(np.argmax(cum_energy >= threshold) + 1)
    share_kept = cum_energy[k - 1] * 100.0

    # 6. Reconstruct rank-k
    img_k = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]

    # 7. Actual reconstruction error
    norm_err = np.linalg.norm(matrix - img_k)

    # 8. Theoretical error from discarded singular values
    dropped_err = np.sqrt(np.sum(S[k:]**2))
    matches = np.isclose(norm_err, dropped_err)

    return k, share_kept, norm_err, dropped_err, matches

# --- Test 1: Zero Matrix Edge Case ---
k0, s0, err0, drop0, match0 = svd_compress(np.zeros((20, 20)))
print(f"Test 1 (Zeros): k={k0}, Share={s0:.1f}%, Actual Err={err0:.4f}, Matches={match0}")

# --- Test 2: Task Image ---
k_task, s_task, err_task, drop_task, match_task = svd_compress(img, threshold=0.90)
print(f"Test 2 (Task Img): k={k_task}, Share={s_task:.2f}%, Actual Err={err_task:.4f}, Theo Err={drop_task:.4f}, Matches Eckart-Young={match_task}")