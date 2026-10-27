import numpy as np

x1 = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
y1 = np.array([18, 25, 33, 39, 48, 56, 63, 72], dtype=float)

x_mean = np.mean(x1)
y_mean = np.mean(y1)
beta1 = np.sum((x1 - x_mean) * (y1 - y_mean)) / np.sum((x1 - x_mean) ** 2)
beta0 = y_mean - beta1 * x_mean

pred_5_5 = beta0 + beta1 * 5.5
pred_10 = beta0 + beta1 * 10.0
y_hat1 = beta0 + beta1 * x1

mse1 = np.mean((y1 - y_hat1) ** 2)
r2_1 = 1 - np.sum((y1 - y_hat1) ** 2) / np.sum((y1 - y_mean) ** 2)

print("Q1:")
print(f"beta0 = {beta0:.3f}, beta1 = {beta1:.3f}")
print(f"Predict x=5.5: {pred_5_5:.3f}")
print(f"Predict x=10:  {pred_10:.3f}")
print(f"MSE = {mse1:.3f}")
print(f"R^2 = {r2_1:.3f}")

x2 = np.array([1, 2, 3, 4, 5, 6], dtype=float)
y2 = np.array([5.2, 7.1, 10.4, 14.8, 20.5, 27.9], dtype=float)
n2 = len(x2)

X_deg1 = np.column_stack([np.ones(n2), x2])
w_deg1 = np.linalg.inv(X_deg1.T @ X_deg1) @ X_deg1.T @ y2
mse_deg1 = np.mean((y2 - X_deg1 @ w_deg1) ** 2)

X_deg2 = np.column_stack([np.ones(n2), x2, x2**2])
w_deg2 = np.linalg.inv(X_deg2.T @ X_deg2) @ X_deg2.T @ y2
mse_deg2 = np.mean((y2 - X_deg2 @ w_deg2) ** 2)

beta = 0.0
alpha = 0.01
for _ in range(100):
    grad = - (2.0 / n2) * np.sum(x2 * (y2 - beta * x2))
    beta -= alpha * grad
loss = np.mean((y2 - beta * x2) ** 2)

print("\nQ2:")
print(f"Degree-1 MSE: {mse_deg1:.3f}")
print(f"Degree-2 MSE: {mse_deg2:.3f}")
print(f"Gradient descent: beta = {beta:.3f}, Loss = {loss:.3f}")

X3 = np.array([
    [0.2, 0.1], [0.5, 0.2], [0.7, 0.4], [1.0, 0.7],
    [1.3, 0.9], [1.5, 1.1], [1.8, 1.4], [2.0, 1.7]
])
y3 = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=float)

X3_b = np.column_stack([np.ones(len(X3)), X3])
w3 = np.zeros(3)

for _ in range(1000):
    z = X3_b @ w3
    p = 1.0 / (1.0 + np.exp(-z))
    w3 -= 0.2 * (X3_b.T @ (p - y3)) / len(y3)

x_test = np.array([1.0, 1.2, 0.8])
p_test = 1.0 / (1.0 + np.exp(-np.dot(x_test, w3)))

probs3 = 1.0 / (1.0 + np.exp(-(X3_b @ w3)))
preds_05 = (probs3 >= 0.5).astype(int)
preds_03 = (probs3 >= 0.3).astype(int)

print("\nQ3:")
print(f"beta0 = {w3[0]:.3f}, beta1 = {w3[1]:.3f}, beta2 = {w3[2]:.3f}")
print(f"x=[1.2, 0.8] p = {p_test:.3f}")
print(f"Threshold 0.5: acc = {np.mean(preds_05 == y3):.3f}, preds = {preds_05}")
print(f"Threshold 0.3: acc = {np.mean(preds_03 == y3):.3f}, preds = {preds_03}")

x4 = np.array([1.0, 2.0, 1.5])
y4 = 1.0
beta4 = np.array([0.1, -0.2, 0.3])

z4 = np.dot(beta4, x4)
p4 = 1.0 / (1.0 + np.exp(-z4))
loss4 = - (y4 * np.log(p4) + (1.0 - y4) * np.log(1.0 - p4))

grad4 = (p4 - y4) * x4
beta4_new = beta4 - 0.1 * grad4

z4_new = np.dot(beta4_new, x4)
p4_new = 1.0 / (1.0 + np.exp(-z4_new))
loss4_new = - (y4 * np.log(p4_new) + (1.0 - y4) * np.log(1.0 - p4_new))

print("\nQ4:")
print(f"z = {z4:.3f}, p = {p4:.3f}, loss = {loss4:.3f}")
print(f"grad = {np.round(grad4, 3)}")
print(f"new beta = {np.round(beta4_new, 3)}")
print(f"new z = {z4_new:.3f}, new p = {p4_new:.3f}, new loss = {loss4_new:.3f}")

Z5 = np.array([
    [1.2, 2.1, 0.4],
    [2.4, 0.8, 0.5],
    [0.3, 1.1, 2.2],
    [1.5, 1.4, 1.3]
])

shift5 = Z5 - np.max(Z5, axis=1, keepdims=True)
exp5 = np.exp(shift5)
P5 = exp5 / np.sum(exp5, axis=1, keepdims=True)

Z5_const = Z5 + 10.0
shift5_const = Z5_const - np.max(Z5_const, axis=1, keepdims=True)
P5_const = np.exp(shift5_const) / np.sum(np.exp(shift5_const), axis=1, keepdims=True)

print("\nQ5:")
print("Softmax probabilities:\n", np.round(P5, 3))
print("Argmax classes:", np.argmax(P5, axis=1))
print("Row sums:", np.round(np.sum(P5, axis=1), 3))
print("Difference (+10):", np.max(np.abs(P5 - P5_const)))

X6 = np.array([
    [0.9, 0.2], [1.1, 0.3], [0.8, 0.4],
    [1.4, 0.8], [1.6, 0.9], [1.5, 0.7],
    [0.4, 1.5], [0.5, 1.7], [0.6, 1.4]
])
y6 = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

W6 = np.zeros((2, 3))
b6 = np.zeros(3)
Y6_onehot = np.eye(3)[y6]

for _ in range(500):
    scores = X6 @ W6 + b6
    shift = scores - np.max(scores, axis=1, keepdims=True)
    probs6 = np.exp(shift) / np.sum(np.exp(shift), axis=1, keepdims=True)
    W6 -= 0.1 * (X6.T @ (probs6 - Y6_onehot)) / len(y6)
    b6 -= 0.1 * np.mean(probs6 - Y6_onehot, axis=0)

preds6 = np.argmax(probs6, axis=1)
sample_losses = -np.log(probs6[np.arange(len(y6)), y6])

print("\nQ6:")
print("Predictions:", preds6)
print("Probabilities:\n", np.round(probs6, 3))
print("Sample losses:", np.round(sample_losses, 3))
print(f"Mean log loss: {np.mean(sample_losses):.3f}")
print("W shape:", W6.shape)
print("W:\n", np.round(W6, 3))
print("b:\n", np.round(b6, 3))
