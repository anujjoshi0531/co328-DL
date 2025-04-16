# **Neural Network from Scratch using NumPy for MNIST**

## **Overview**

This project implements a **fully connected Artificial Neural Network (ANN) from scratch** using **NumPy** to classify digits from the **MNIST dataset**. No deep learning libraries like TensorFlow or PyTorch are used, making this a great educational tool for understanding how neural networks work at a fundamental level.

---

## **Features**

- **Custom implementation of forward and backward propagation**
- **Gradient Descent optimization**
- **Cross-entropy loss function**
- **NumPy-based matrix operations for efficiency**
- **Configurable number of layers and neurons**
- **Handwritten digit classification from the MNIST dataset**

---

## **Dataset**

The **MNIST dataset** consists of **70,000 images** of handwritten digits (0-9), each of size **28×28 pixels**:

- **Training Set**: 60,000 images
- **Test Set**: 10,000 images
- **Format**: Grayscale images (values between 0 and 255)

---

## **Installation & Setup**

### **1. Clone the repository:**

```bash
git clone https://github.com/yourusername/mnist-ann-numpy.git
cd mnist-ann-numpy
```

### **2. Install dependencies:**

Ensure you have Python installed, then install NumPy and other required libraries:

```bash
pip install numpy matplotlib
```

## **Neural Network Architecture**

- **Input Layer**: 784 neurons (28×28 pixels flattened)
- **Hidden Layers**: Configurable, typically 1-2 layers with 128/64 neurons each
- **Activation Function**: ReLU for hidden layers, Softmax for output
- **Output Layer**: 10 neurons (one for each digit 0-9)

---

## **Training Details**

- **Loss Function**: Cross-entropy
- **Optimizer**: Gradient Descent (or variants like Adam, SGD)
- **Learning Rate**: Configurable (default: 0.01)
- **Batch Size**: Configurable (default: 32)
- **Epochs**: Configurable (default: 10)

---

## **Results**

- Expected accuracy: **95-98%** on the MNIST test dataset
- **Loss and accuracy plots** can be visualized using Matplotlib

---

## **Contributing**

Feel free to fork the repository and contribute improvements. You can:

- Optimize training performance
- Add support for different activation functions
- Implement additional optimizers

---

## **License**

This project is open-source and licensed under the **MIT License**.

---

## **Acknowledgments**

- The MNIST dataset is provided by **Yann LeCun et al.**
- Inspired by classic implementations of neural networks from scratch.
