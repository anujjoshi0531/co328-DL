# Deep Learning Experiments using Pytorch

This repository contains implementation code and experiment reports for various foundational and advanced deep learning tasks completed as part of the Deep Learning course (CO328) at Delhi Technological University (2024–25).

## 🧠 Table of Contents

- [Overview](#overview)
- [Experiments](#experiments)
- [Installation](#installation)
- [Usage](#usage)
- [Results Summary](#results-summary)
- [Contributors](#contributors)

---

## 📘 Overview

The project includes 8 experiments covering both classical and cutting-edge deep learning topics:

- Basic tensor operations with PyTorch and NumPy
- Feedforward Neural Networks from scratch
- CNNs for image classification
- RNNs for text generation
- LSTMs with Attention for machine translation

All models are built using Python, with minimal use of high-level libraries to emphasize core learning concepts.

---

## 🧪 Experiments

### 1. Fundamentals of Tensor Operations

- Frameworks: PyTorch, NumPy
- Topics: Tensor creation, indexing, broadcasting, in-place operations

### 2. Feedforward Neural Network from Scratch (MNIST)

- Built using NumPy only
- Features: He/Xavier Initialization, Cross-Entropy loss, SGD
- Accuracy: **89.47%** on MNIST

### 3. Linear vs Non-linear Data Classification

- ReLU effect on linear vs circular datasets
- Accuracy: ~99% with ReLU on circular data

### 4. CNN for Image Classification

- Datasets: Cats vs Dogs, CIFAR-10
- Techniques: Data Augmentation, Weight Init, Activation/Optimizer tuning
- Best Accuracies:
  - Dogs vs Cats: **74.68%**
  - CIFAR-10: **63.05%**

### 5. ResNet-18 on CIFAR-10

- Residual Learning & Batch Norm
- Accuracy: **91%** on test set

### 6. Poem Generation with RNN, LSTM, GRU

- Input Encodings: One-hot vs Embedding
- Best Result: **LSTM + Embedding**, Accuracy: **30.56%**

### 7. English-to-Spanish Translation (Seq2Seq)

- Architecture: LSTM Encoder-Decoder without attention
- BLEU Score: **0.0878**

### 8. English-to-Spanish Translation with Attention

- Mechanisms: Bahdanau & Luong
- BLEU Scores: Bahdanau – **0.0334**, Luong – **0.0433**

---

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/anujjoshi3105/DL-Pytorch.git
cd DTU-Deep-Learning-Experiments

# Install dependencies
pip install -r requirements.txt
```

> ⚠️ Each experiment is in its own folder with independent requirements.

---

## 🚀 Usage

For Jupyter Notebooks:

```bash
jupyter notebook
```

---

## 📊 Results Summary

| Experiment | Dataset                 | Best Accuracy | Model Type       |
| ---------- | ----------------------- | ------------- | ---------------- |
| Exp 2      | MNIST                   | 89.47%        | NN from Scratch  |
| Exp 3      | Circular                | 99.0%         | ReLU vs No-ReLU  |
| Exp 4      | Dogs vs Cats            | 74.68%        | CNN              |
| Exp 5      | CIFAR-10                | 90.0%         | ResNet-18        |
| Exp 6      | Poem Gen                | 30.56%        | LSTM + Embedding |
| Exp 7      | Translation             | BLEU 0.0878   | LSTM Seq2Seq     |
| Exp 8      | Translation + Attention | BLEU 0.0433   | Luong Attention  |

---

## 👨‍💻 Contributors

- **Anuj Joshi** – [anujjoshicode](https://github.com/anujjoshi3105)  
  Student, B.Tech CSE, Delhi Technological University  
  Roll No: 2K22/CO/74

- **Supervisor** – Prof. Anil Singh Parihar  
  Department of Computer Science & Engineering

---

## 📎 License

This project is for educational use only.
