import numpy as np
 
# ── Load & shuffle ────────────────────────────────────────────────────────────
data = np.genfromtxt('students.csv', delimiter=',', skip_header=1)
np.random.seed(42)
data = data[np.random.permutation(len(data))]   # shuffle before splitting
 
X = data[:, :3]
Y = data[:, 3].reshape(-1, 1)
 
# ── Normalize (fit on train only — no leakage) ────────────────────────────────
n = len(X)
train_end = int(0.70 * n)
val_end   = int(0.85 * n)
 
X_train, Y_train = X[:train_end],          Y[:train_end]
X_val,   Y_val   = X[train_end:val_end],   Y[train_end:val_end]
X_test,  Y_test  = X[val_end:],            Y[val_end:]
 
x_min = X_train.min(axis=0)
x_max = X_train.max(axis=0)
 
def normalize(arr):
    return (arr - x_min) / (x_max - x_min + 1e-8)   # 1e-8 avoids div-by-zero
 
X_train = normalize(X_train)
X_val   = normalize(X_val)
X_test  = normalize(X_test)
 
# ── Weight initialisation ─────────────────────────────────────────────────────
def init_weights():
    w1 = np.random.randn(3, 4) * 0.01
    w2 = np.random.randn(4, 1) * 0.01
    b1 = np.zeros((1, 4))
    b2 = np.zeros((1, 1))
    return w1, b1, w2, b2
 
# ── Activations ───────────────────────────────────────────────────────────────
def relu(z):
    return np.maximum(0, z)
 
def relu_grad(z):
    return (z > 0).astype(float)
 
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # clip prevents overflow
 
# ── Forward pass (no globals) ─────────────────────────────────────────────────
def forward(X, w1, b1, w2, b2):
    z1     = X @ w1 + b1
    hidden = relu(z1)
    z2     = hidden @ w2 + b2
    output = sigmoid(z2)
    cache  = (z1, hidden, z2)
    return output, cache
 
# ── Binary cross-entropy loss ─────────────────────────────────────────────────
def compute_loss(Y, output):
    eps = 1e-8
    return -np.mean(Y * np.log(output + eps) + (1 - Y) * np.log(1 - output + eps))
 
# ── Backward pass (no globals) ────────────────────────────────────────────────
def backward(X, Y, output, cache, w2):
    z1, hidden, _ = cache
    m = X.shape[0]
 
    # Output layer — BCE + sigmoid gradient simplifies to (output - Y)
    d_out = (output - Y) / m
 
    dw2 = hidden.T @ d_out
    db2 = np.sum(d_out, axis=0, keepdims=True)
 
    # Hidden layer — ReLU gradient
    d_hidden = (d_out @ w2.T) * relu_grad(z1)
 
    dw1 = X.T @ d_hidden
    db1 = np.sum(d_hidden, axis=0, keepdims=True)
 
    return dw1, db1, dw2, db2
 
# ── Weight update────────────────────────────────────────────────
def update(w1, b1, w2, b2, dw1, db1, dw2, db2, lr):
    w1 = w1 - lr * dw1
    w2 = w2 - lr * dw2
    b1 = b1 - lr * db1
    b2 = b2 - lr * db2
    return w1, b1, w2, b2
 
# ── Training loop ─────────────────────────────────────────────────────────────
EPOCHS        = 1000
LEARNING_RATE = 0.1
 
w1, b1, w2, b2 = init_weights()
 
for epoch in range(EPOCHS):
    output, cache       = forward(X_train, w1, b1, w2, b2)
    train_loss          = compute_loss(Y_train, output)
    dw1, db1, dw2, db2 = backward(X_train, Y_train, output, cache, w2)
    w1, b1, w2, b2      = update(w1, b1, w2, b2, dw1, db1, dw2, db2, LEARNING_RATE)
 
    if epoch % 100 == 0:
        val_out, _  = forward(X_val, w1, b1, w2, b2)
        val_loss    = compute_loss(Y_val, val_out)
        print(f"Epoch {epoch:4d} | train loss: {train_loss:.4f} | val loss: {val_loss:.4f}")
 
# ── Evaluation ────────────────────────────────────────────────────────────────
def predict(X, w1, b1, w2, b2):
    output, _ = forward(X, w1, b1, w2, b2)
    return (output >= 0.5).astype(int)
 
predictions = predict(X_test, w1, b1, w2, b2)
accuracy    = np.mean(predictions == Y_test) * 100
print(f"\nTest accuracy: {accuracy:.2f}%")


