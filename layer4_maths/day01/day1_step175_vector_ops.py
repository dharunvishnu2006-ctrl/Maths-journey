import numpy as np

def magnitude(v):
    v = np.asarray(v, dtype=float)
    return float(np.sqrt(np.sum(v**2)))


def add(u, v):
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    if u.shape != v.shape:
        raise ValueError(f"Cannot add vectors of shapes {u.shape} and {v.shape}.")
    return u + v


def scale(c, v):
    v = np.asarray(v, dtype=float)
    return c * v


def dot_from_scratch(u, v):
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    if u.shape != v.shape:
        raise ValueError(f"Cannot dot vectors of shapes {u.shape} and {v.shape}.")
    return float(np.sum(u * v))


def cos_sim(u, v):
    mag_u = magnitude(u)
    mag_v = magnitude(v)
    if np.isclose(mag_u, 0.0) or np.isclose(mag_v, 0.0):
        raise ValueError("Cosine similarity is undefined for a zero vector.")
    return dot_from_scratch(u, v) / (mag_u * mag_v)

a = np.array([2.0, 3.0])
b = np.array([4.0, -1.0])
p = np.array([1.0, 2.0])
q = np.array([-2.0, 1.0])

print("a + b:", a + b)
print("3 * a:", 3 * a)
print("a * b:", a * b)

dot_mine = np.sum(a * b)
dot_lib = np.dot(a, b)
print("dot (mine):", dot_mine)
print("dot (np.dot):", dot_lib, "| a @ b:", a @ b)
print("Match?", np.allclose(dot_mine, dot_lib))

cos_ab = np.dot(a, b) / (magnitude(a) * magnitude(b))
print("cos_sim(a, b):", cos_ab)

angle_a = np.arctan2(a[1], a[0])
angle_b = np.arctan2(b[1], b[0])
theta = angle_a - angle_b
print("theta (degrees):", np.degrees(theta))
print(
    "Geometric check:",
    np.allclose(dot_lib, magnitude(a) * magnitude(b) * np.cos(theta)),
)

print("p . q:", np.dot(p, q))
angle_pq = np.degrees(np.arccos(np.clip(np.dot(p, q) / (magnitude(p) * magnitude(q)), -1.0, 1.0)))
print("Angle between p and q:", angle_pq, "degrees")

bad_pairs = [([1, 2, 3], [10, 20]), ([1, 2, 3], [10])]
for u, v in bad_pairs:
    try:
        add(u, v)
        print(u, "+", v, "-> no error (BUG!)")
    except ValueError as e:
        print(u, "+", v, "-> caught:", e)

print("add([2, 3], [4, -1]):", add([2, 3], [4, -1]))
print("scale(3, [2, 3]):", scale(3, [2, 3]))
print("scale(0, [2, 3]):", scale(0, [2, 3]))
print("scale(-1, [2, 3]):", scale(-1, [2, 3]))

try:
    cos_sim([0, 0], [1, 2])
except ValueError as e:
    print("Zero vector caught:", e)

try:
    dot_from_scratch([1, 2, 3], [1, 2])
except ValueError as e:
    print("Shape mismatch caught:", e)

pairs = [
    ([1, 0], [0, 1]),
    ([1, 0], [1, 0]),
    ([1, 0], [-1, 0]),
    ([1, 2, 3], [2, 4, 6]),
]
for u, v in pairs:
    mine = dot_from_scratch(u, v)
    lib = np.dot(u, v)
    print(
        u,
        v,
        "| dot:",
        mine,
        "| matches np.dot:",
        np.allclose(mine, lib),
        "| cos_sim:",
        round(cos_sim(u, v), 4),
    )


def find_twin(live, known):
    live = np.asarray(live, dtype=float)
    if live.size == 0 or np.isclose(magnitude(live), 0.0) or not known:
        return (None, -1.0)

    best_name, best_score = None, float("-inf")
    for name, sig in known.items():
        sig = np.asarray(sig, dtype=float)
        if sig.shape != live.shape or np.isclose(magnitude(sig), 0.0):
            continue

        score = float(np.clip(cos_sim(live, sig), -1.0, 1.0))
        if score > best_score:
            best_score, best_name = score, name

    return (best_name, best_score if best_name is not None else -1.0)


print("\n--- find_twin Tests ---")
live_attack = [4, 20, 2]
known_attacks = {
    "port_scan": [8, 40, 4],
    "brute_force": [2, 2, 50],
    "ddos": [1, 200, 0],
    "data_exfil": [9, 5, 1],
}

print("1. Empty known dict       :", find_twin(live_attack, {}))

print("2. Zero vector live       :", find_twin([0, 0, 0], known_attacks))

bad_known = {
    "zero_sig": [0, 0, 0],
    "wrong_shape": [10, 20],
    "port_scan": [8, 40, 4],
}
print("3. Bad signatures skipped :", find_twin(live_attack, bad_known))
print("4. Full problem run       :", find_twin(live_attack, known_attacks))