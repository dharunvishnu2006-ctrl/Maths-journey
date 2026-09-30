import numpy as np

A = np.array([[1, 2], [3, 4]])
print("A:\n", A)
print("A shape:", A.shape)

zeros_mat = np.zeros((2, 3))
print("\nzeros_mat (2, 3):\n", zeros_mat)

I = np.identity(2)
print("\nIdentity I_2:\n", I)
print("I @ A:\n", I @ A)
print("Is I @ A equal to A?", np.allclose(I @ A, A))

flat_v = np.array([1, 2, 3])
row_v = np.array([[1, 2, 3]])
col_v = np.array([[1], [2], [3]])
print(
    f"\nShapes -> flat: {flat_v.shape}, row: {row_v.shape}, col: {col_v.shape}"
)

demo_events = np.array([[2, 25, 1], [8, 40, 6], [0, 200, 0]])
print(
    f"Single index shape: {demo_events[1].shape} | Slice shape: {demo_events[1:2].shape}"
)


def describe(M):
    M = np.asarray(M)
    is_square = M.ndim == 2 and M.shape[0] == M.shape[1]
    print(
        f"shape={M.shape} | ndim={M.ndim} | size={M.size} | "
        f"dtype={M.dtype} | is_square={is_square}"
    )


print("\n--- Task 1 Tests ---")
describe([[1, 2], [3, 4]]) 
describe([[1, 2], [3, 4], [5, 6]]) 
describe(np.identity(4))  
describe([1, 2, 3])  
describe(np.zeros((1, 5)))  


def stack_events(events_list):
    if not events_list:
        raise ValueError("events_list cannot be empty.")
    first_len = len(events_list[0])
    for idx, e in enumerate(events_list):
        if len(e) != first_len:
            raise ValueError(
                f"Row {idx} has length {len(e)}, expected {first_len}."
            )
    return np.asarray(events_list, dtype=float)


def feature_column(M, j):
    M = np.asarray(M)
    if M.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got ndim={M.ndim}.")
    if not (0 <= j < M.shape[1]):
        raise IndexError(
            f"Column index {j} out of bounds for matrix with {M.shape[1]} columns."
        )
    return M[:, j]


print("\n--- Task 2 Edge Cases ---")
try:
    stack_events([])
except ValueError as e:
    print(f"1. Empty list caught -> {e}")

try:
    stack_events([[1, 2, 3], [4, 5]])
except ValueError as e:
    print(f"2. Ragged lengths caught -> {e}")

try:
    feature_column(np.array([1, 2, 3]), 0)
except ValueError as e:
    print(f"3. Flat vector caught -> {e}")

try:
    feature_column([[1, 2], [3, 4]], 99)
except IndexError as e:
    print(f"4. Out of range caught -> {e}")

print("\n--- Task 2 Normal Case ---")
cloudshield_events = [
    [2, 25, 1],  
    [8, 40, 6],  
    [0, 200, 0],  
    [5, 38, 5], 
    [4, 22, 1],  
]

M = stack_events(cloudshield_events)
col_0 = feature_column(M, 0)

print("Stacked Matrix Shape:", M.shape)
print("Matrix:\n", M)
print("Col 0 (severity):", col_0)


def scale_report(M, feature_names):
    M = np.asarray(M, dtype=float)

    if M.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got ndim={M.ndim}.")
    if M.shape[0] == 0:
        raise ValueError("Matrix cannot be empty (0 rows).")
    if len(feature_names) != M.shape[1]:
        raise ValueError(
            f"Expected {M.shape[1]} feature names for {M.shape[1]} columns, "
            f"got {len(feature_names)}."
        )

    mins = M.min(axis=0)
    maxs = M.max(axis=0)
    means = M.mean(axis=0)
    ranges = maxs - mins
    max_r = ranges.max()

    if max_r == 0.0:
        dominances = np.zeros_like(ranges)
    else:
        dominances = ranges / max_r

    report = []
    for j in range(M.shape[1]):
        report.append(
            {
                "name": feature_names[j],
                "min": float(mins[j]),
                "max": float(maxs[j]),
                "mean": float(means[j]),
                "range": float(ranges[j]),
                "dominance": float(dominances[j]),
            }
        )

    return report

print("--- Edge Case Tests ---")

try:
    scale_report(np.empty((0, 3)), ["f1", "f2", "f3"])
except ValueError as e:
    print(f"1. Empty matrix caught -> {e}")

try:
    scale_report([[1, 2, 3]], ["col_a", "col_b"])
except ValueError as e:
    print(f"2. Name mismatch caught -> {e}")

try:
    scale_report([1, 2, 3], ["col_a", "col_b", "col_c"])
except ValueError as e:
    print(f"3. Flat vector caught -> {e}")

identical_rows = [
    [10.0, 50.0, 1.0],
    [10.0, 50.0, 1.0],
    [10.0, 50.0, 1.0],
]
identical_report = scale_report(
    identical_rows, ["severity", "requests", "logins"]
)
print("\n4. All-identical rows (zero range test):")
for item in identical_report:
    print(
        f"   {item['name']}: range={item['range']}, dominance={item['dominance']}"
    )

print("\n--- Normal Case: CloudShield Events ---")
cloudshield_events = [
    [2, 25, 1],
    [8, 40, 6],
    [0, 200, 0],
    [5, 38, 5],
    [4, 22, 1],
]
feature_names = ["severity", "requests_per_minute", "failed_logins"]

report = scale_report(cloudshield_events, feature_names)
for r in report:
    print(
        f"{r['name']:<22} | min={r['min']:>5.1f} | max={r['max']:>5.1f} | "
        f"mean={r['mean']:>5.1f} | range={r['range']:>5.1f} | dominance={r['dominance']:.3f}"
    )