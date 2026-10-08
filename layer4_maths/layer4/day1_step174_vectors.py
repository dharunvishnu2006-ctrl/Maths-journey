import numpy as np

temperature = 32.0
print("Scalar:", temperature)

v = np.array([6.0, 8.0])
print("Vector:", v)
print("Shape:", v.shape)

magnitude_mine = np.sqrt(np.sum(v ** 2))
print("My magnitude:", magnitude_mine)

magnitude_lib = np.linalg.norm(v)
print("Library magnitude:", magnitude_lib)

print("Match?", np.allclose(magnitude_mine, magnitude_lib))

v_hat = v / magnitude_mine
print("Unit vector:", v_hat)

print("Length of unit vector:", np.linalg.norm(v_hat))
print("Is it 1?", np.allclose(np.linalg.norm(v_hat), 1.0))


def magnitude(v):
    v = np.array(v, dtype=float)   
    squares = v ** 2               
    total = np.sum(squares)        
    return np.sqrt(total)          

test_vectors = [[0, 0], [3, 4], [1, 2, 2], [-5, 12], [1, 1, 1, 1]]

for vec in test_vectors:
    mine = magnitude(vec)
    lib = np.linalg.norm(vec)
    print(vec, "mine:", mine, "library:", lib, "match:", np.allclose(mine, lib))


def normalize(v):

    v = np.array(v, dtype=float)
    mag = magnitude(v)                        
    if np.isclose(mag, 0.0):               
        raise ValueError("Cannot normalize a zero vector: it has no direction.")
    return v / mag                            


try:
    normalize([0, 0])
except ValueError as e:
    print("Zero vector caught:", e)

for vec in [[3, 4], [1, 2, 2], [-5, 12], [1, 1, 1, 1]]:
    unit = normalize(vec)
    print(vec, "->", unit, "| length:", magnitude(unit),
          "| is 1:", np.allclose(magnitude(unit), 1.0))    


def most_intense(events):
    if not events:
        return -1

    best_idx = -1
    max_mag = -1.0

    for i, ev in enumerate(events):
        mag = magnitude(np.array(ev))
        if mag > max_mag:
            max_mag = mag
            best_idx = i

    return best_idx

test_cases = [
    ("Empty list", [], -1),
    ("All zero vectors", [[0, 0, 0], [0, 0, 0]], 0),
    ("Single event", [[5, 12, 0]], 0),
    ("Tie for max", [[3, 4, 0], [4, 3, 0]], 0),
    ("Problem events", [[2, 10, 0], [8, 40, 6], [0, 0, 0], [5, 60, 1]], 3),
]

for name, evs, expected in test_cases:
    actual = most_intense(evs)
    print(
        f"{name:<18} | Expected: {expected:>2} | Got: {actual:>2} | Pass: {actual == expected}"
    )          
print(magnitude([3, 4]))    