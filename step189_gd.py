def loss(w):
    return (w - 3) ** 2

def grad(w):
    return 2 * (w - 3)

w = 0.0
alpha = 0.1
for step in range(1, 6):
    w = w - alpha * grad(w)      # the update rule
    print(step, round(w, 4), round(loss(w), 4))

for alpha in [0.001, 0.01, 0.1, 1.1]:
    w = 0.0
    for step in range(50):
        w = w - alpha * grad(w)
    print(alpha, round(w, 4), round(loss(w), 4))    

def loss2(w):
    return w**4 - 2*w**2 + 0.5*w

def grad2(w):
    return 4*w**3 - 4*w + 0.5

for start in [2.0, -2.0]:
    w = start
    for step in range(200):
        w = w - 0.05 * grad2(w)
    print(start, round(w, 4), round(loss2(w), 4))    