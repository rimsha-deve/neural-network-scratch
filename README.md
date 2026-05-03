# Neural Network from Scratch — NumPy Only
![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-Manual%20Implementation-orange.svg)
![Project](https://img.shields.io/badge/Status-From%20Scratch-green.svg)
![ML](https://img.shields.io/badge/Machine%20Learning-Neural%20Network-red.svg)

> A fully functional binary classification neural network built using **only Python and NumPy** — no TensorFlow, no PyTorch, no sklearn.

---

## What This Is

Most people learn neural networks by calling `model.fit()`. I built one from the ground up — every forward pass, every gradient, every weight update written by hand. This project is proof that I understand what's actually happening inside the black box.

---

## How It Works

```
Input Layer (3 features)
        ↓
Hidden Layer (4 neurons) — ReLU activation
        ↓
Output Layer (1 neuron) — Sigmoid activation → Binary prediction
```

**Training pipeline:**
1. Forward pass — compute predictions
2. Binary cross-entropy loss — measure error
3. Backpropagation — compute gradients via chain rule (by hand)
4. Gradient descent — update weights

---

## Technical Highlights

| What | Detail |
|---|---|
| Language | Python 3 |
| Libraries | NumPy only |
| Architecture | 3 → 4 → 1 fully connected |
| Activation (hidden) | ReLU |
| Activation (output) | Sigmoid |
| Loss function | Binary cross-entropy |
| Optimizer | Gradient descent |
| Data split | 70% train / 15% val / 15% test |

---

## Key Engineering Decisions

- **No data leakage** — normalization is fit on training data only, then applied to val/test
- **Validation monitoring** — val loss tracked every epoch to detect overfitting early
- **No global state** — weights are passed explicitly into every function, making the code modular and testable
- **Numerically stable sigmoid** — input clipping prevents overflow on extreme values

---

## Project Structure

```
neural-network-scratch/
├── 01_neural_net.py    # Full implementation
├── students.csv        # Dataset
└── README.md
```

---

## Run It Yourself

```bash
git clone https://github.com/rimsha-deve/neural-network-scratch.git
cd neural-network-scratch
pip install numpy
python 01_neural_net.py
```

**Expected output:**
```
Epoch    0 | train loss: 0.6931 | val loss: 0.6924
Epoch  100 | train loss: 0.4821 | val loss: 0.4903
...
Test accuracy: XX.XX%
```

---

## What I Learned

Building this without libraries forced me to deeply understand:
- How gradients flow backward through a network (chain rule in practice)
- Why activation function choice matters (ReLU vs sigmoid and vanishing gradients)
- What data leakage actually is and how to prevent it
- The difference between training loss and validation loss

---

## What's Next

- [ ] Add momentum / Adam optimizer
- [ ] Extend to multi-class classification (softmax output)
- [ ] Add support for variable hidden layer sizes
- [ ] Implement mini-batch gradient descent

---

*Built as part of my AI/ML engineering learning path.*
