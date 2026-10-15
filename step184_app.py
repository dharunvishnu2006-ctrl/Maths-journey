import numpy as np

def run_fare_model(km_data, fares_data, label="Dataset"):
    print(f"\n--- {label} ---")
    km = np.array(km_data, dtype=float)
    b = np.array(fares_data, dtype=float)
    A = np.column_stack([np.ones_like(km), km])

    params, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
    base, rate = params[0], params[1]
    print(f"Base: ₹{base:.2f}, Rate: ₹{rate:.2f}/km | Rank: {rank}")

    fare_8km = np.array([1.0, 8.0]) @ params
    print(f"8 km Fare: ₹{fare_8km:.2f}")

    sse_opt = np.sum((b - A @ params) ** 2)
    sse_pert = np.sum((b - A @ [base + 1.0, rate]) ** 2)
    print(f"SSE opt: {sse_opt:.4f} < SSE perturbed: {sse_pert:.4f} -> {sse_opt < sse_pert}")

    cond_A = np.linalg.cond(A)
    print(f"cond(A): {cond_A:.2f}")
    if cond_A > 1000:
        print("WARNING: cond(A) > 1000! System is ill-conditioned.")

# Edge case & Real run
run_fare_model([5, 5, 5, 5, 5], [48, 55, 73, 88, 115], label="Edge Case: All 5 km")
run_fare_model([2, 3, 5, 7, 10], [48, 55, 73, 88, 115], label="Real Data")