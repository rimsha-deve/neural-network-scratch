# 🧠 Neural Network from Scratch (NumPy Only)

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-Manual%20Implementation-orange.svg)
![Project](https://img.shields.io/badge/Status-From%20Scratch-green.svg)
![ML](https://img.shields.io/badge/Machine%20Learning-Neural%20Network-red.svg)

> Built to demonstrate deep understanding of neural networks by implementing forward and backward propagation completely from scratch using only NumPy.

---

# 🚀 Why this project

- How neural networks learn step by step  
- How gradients flow backward using chain rule  
- How weights are updated using gradient descent  
- How loss decreases during training  

---

# 🧠 Model Architecture

Input Layer → Hidden Layer (ReLU) → Output Layer (Sigmoid) → Binary Classification

---

# ⚙️ Training Pipeline

- Load dataset  
- Shuffle data  
- Split into train/val/test  
- Normalize (train only)  
- Forward propagation  
- Loss calculation  
- Backpropagation  
- Gradient descent update  

---

# ⭐ Highlights

- Built from scratch (no ML libraries)
- Manual backpropagation
- Fully vectorized NumPy code
- No data leakage
- Stable sigmoid implementation

---

# 🔑 Core Forward Pass

```python
z1 = X @ w1 + b1
a1 = np.maximum(0, z1)
🔙 Backpropagation
dL/dz = output - Y
📊 Results
| Metric         | Value |
| -------------- | ----- |
| Train Accuracy | XX%   |
| Test Accuracy  | XX%   |
| Loss           | X.XX  |
📁 Project Structure
neural-network-scratch/
├── 01_neural_net.py
├── students.csv
└── README.md
▶️ How to Run
git clone https://github.com/YOUR_USERNAME/neural-network-scratch.git
cd neural-network-scratch

pip install numpy
python 01_neural_net.py
📈 Output Example
Epoch 0   loss: 0.6931
Epoch 100 loss: 0.5123
Epoch 900 loss: 0.1204

Test accuracy: 93%
💡 What I Learned
Neural network internals
Backpropagation math
Gradient descent
Data normalization importance
Vectorization benefits
🚀 Future Improvements
Adam optimizer
Softmax classification
Mini-batch training
Loss visualization
🧾 Final Note

Built to understand neural networks from scratch without using any ML frameworks.

z2 = a1 @ w2 + b2
output = 1 / (1 + np.exp(-np.clip(z2, -500, 500)))
