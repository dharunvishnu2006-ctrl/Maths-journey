# All four outcomes
omega = {(0, 0), (0, 1), (1, 0), (1, 1)}

# Sensor events
S1 = {x for x in omega if x[0] == 1}
S2 = {x for x in omega if x[1] == 1}
S3 = {x for x in omega if x[0] != x[1]}

def prob(event):
    return len(event) / len(omega)

def test_independence(A, B):
    return prob(A & B) == prob(A) * prob(B)

print("S1,S2:", test_independence(S1, S2))
print("S1,S3:", test_independence(S1, S3))
print("S2,S3:", test_independence(S2, S3))

triple = S1 & S2 & S3

print("Triple intersection:", triple)
print("Triple probability:", prob(triple))
print("Product:", prob(S1) * prob(S2) * prob(S3))