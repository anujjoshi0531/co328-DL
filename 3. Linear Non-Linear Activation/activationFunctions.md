## **Linear Activation**

1. **Identity Function**: \( f(x) = x \)

---

## **Non-Linear Activation Functions**

### **Sigmoid Family**

1. **Sigmoid**: \( f(x) = \frac{1}{1 + e^{-x}} \)
2. **Softmax**: \( f(x*i) = \frac{e^{x_i}}{\sum*{j} e^{x_j}} \)
3. **Hard Sigmoid**: A piecewise linear approximation of the sigmoid.

### **Tanh Family**

4. **Tanh** (Hyperbolic Tangent): \( f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} \)

---

### **Rectified Linear Unit (ReLU) Family**

5. **ReLU** (Rectified Linear Unit): \( f(x) = \max(0, x) \)
6. **Leaky ReLU**: \( f(x) = x \) if \( x > 0 \), else \( f(x) = \alpha x \), where \( \alpha \) is a small constant.
7. **Parametric ReLU (PReLU)**: Similar to Leaky ReLU, but \( \alpha \) is learned during training.
8. **Exponential Linear Unit (ELU)**: \( f(x) = x \) if \( x > 0 \), else \( f(x) = \alpha (e^x - 1) \).
9. **Scaled Exponential Linear Unit (SELU)**: Self-normalizing activation function.
10. **GELU** (Gaussian Error Linear Unit): Smooth approximation to the ReLU.

---

### **Piecewise Linear Functions**

11. **Maxout**: Outputs the maximum of a set of linear functions.
12. **Softplus**: \( f(x) = \ln(1 + e^x) \), a smooth approximation of ReLU.
13. **Swish**: \( f(x) = x \cdot \text{sigmoid}(x) \), proposed by Google.

---

### **Radial Basis Functions**

14. **Gaussian Activation**: \( f(x) = e^{-(x - c)^2} \), where \( c \) is the center.

---

### **Step Functions**

15. **Binary Step**: \( f(x) = 1 \) if \( x > 0 \), else \( f(x) = 0 \).

---

### **Custom or Advanced Functions**

16. **Mish**: \( f(x) = x \cdot \text{tanh}(\ln(1 + e^x)) \)
17. **Sinusoidal (Sin)**: \( f(x) = \sin(x) \), used in specific tasks.
18. **ArcTan**: \( f(x) = \arctan(x) \)
19. **Bent Identity**: \( f(x) = \frac{\sqrt{x^2 + 1} - 1}{2} + x \)
20. **Softsign**: \( f(x) = \frac{x}{1 + |x|} \)

---

## **Task-Specific Activation Functions**

21. **LogSoftmax**: Combines softmax and logarithmic operations.
22. **Thresholded ReLU**: Similar to ReLU, but activates only if \( x > \text{threshold} \).

---

## **Selection Criteria**

- **Sigmoid** and **Softmax**: Commonly used in classification tasks.
- **ReLU** and variants: Standard for hidden layers in deep learning.
- **Tanh**: Preferred in shallow networks.
- **Mish** and **Swish**: Advanced alternatives for smoother gradients.
