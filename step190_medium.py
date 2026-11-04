import numpy as np
import matplotlib.pyplot as plt

X = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0]])
y = np.array([3.0, 5.0, 7.0, 9.0])
n = X.shape[0]
w = np.zeros(2)
alpha = 0.1
epochs = 1000
losses = []

for epoch in range(epochs):
    pred = X @ w
    error = pred - y
    loss = (error ** 2).sum() / (2 * n)
    losses.append(loss)
    grad = X.T @ error / n
    w = w - alpha * grad

print("first loss:", losses[0])
print("last loss:", losses[-1])
print("final w:", w)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(losses)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("normal y-axis")

plt.subplot(1, 2, 2)
plt.plot(losses)
plt.yscale("log")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("log y-axis")

plt.tight_layout()
plt.savefig("step190_loss_curve.png")
plt.show()