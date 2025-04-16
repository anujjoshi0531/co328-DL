# **CIFAR-10 CNN Classification**

## **Overview**

The **CIFAR-10** dataset is a widely used dataset for **multi-class image classification**. It consists of labeled images across 10 different categories, making it an excellent benchmark for training **convolutional neural networks (CNNs)** to classify various objects.

---

## **Dataset Details**

- **Source**: Created by the **Canadian Institute for Advanced Research (CIFAR)**.
- **Total Images**: 60,000 images (50,000 training, 10,000 test).
- **Image Format**: PNG, RGB.
- **Image Size**: Fixed at **32×32 pixels**.
- **Classes**:
  - **0**: Airplane
  - **1**: Automobile
  - **2**: Bird
  - **3**: Cat
  - **4**: Deer
  - **5**: Dog
  - **6**: Frog
  - **7**: Horse
  - **8**: Ship
  - **9**: Truck

---

## **Challenges**

1. **Low Resolution**: Images are only 32×32 pixels, making object details less distinct.
2. **Class Similarity**: Certain classes (e.g., automobile vs. truck) have overlapping visual features.
3. **Complex Backgrounds**: Some images contain cluttered backgrounds, making classification harder.
4. **Small Dataset**: Compared to datasets like ImageNet, CIFAR-10 is relatively small.

---

## **Preprocessing Steps**

1. **Resizing**: Though images are fixed at **32×32**, they can be upscaled for deep networks.
2. **Normalization**: Scale pixel values to **[0,1]** or **[-1,1]** for stable training.
3. **Data Augmentation**: Apply transformations like **random cropping, flipping, rotation, and brightness adjustments** to improve generalization.
4. **Train-Validation Split**: Typically **80% training, 20% validation**.

---

## **Weight Initialization Techniques**

Proper **weight initialization** prevents gradient issues.

1. **Xavier (Glorot) Initialization** (for sigmoid/tanh activations):
   \[ W \sim U\left(-\frac{\sqrt{6}}{\sqrt{n*{in} + n*{out}}}, \frac{\sqrt{6}}{\sqrt{n*{in} + n*{out}}}\right) \]

2. **He Initialization** (for ReLU activations):
   \[ W \sim \mathcal{N}\left(0, \frac{2}{n\_{in}}\right) \]

3. **LeCun Initialization** (for SELU activation):
   \[ W \sim \mathcal{N}\left(0, \frac{1}{n\_{in}}\right) \]

---

## **Activation Functions**

1. **ReLU (Rectified Linear Unit)**:
   \[ f(x) = \max(0, x) \]

   - Fast convergence, but **dying ReLU problem**.

2. **Leaky ReLU**:
   \[ f(x) = \max(0.01x, x) \]

   - Fixes dying ReLU by allowing small negative values.

3. **Softmax** (for multi-class classification):
   \[ f(x*i) = \frac{e^{x_i}}{\sum*{j} e^{x_j}} \]
   - Converts logits into probability scores.

---

## **Optimizers**

1. **SGD (Stochastic Gradient Descent)**

   - Good for small datasets but slow convergence.

2. **Adam (Adaptive Moment Estimation) [Recommended]**
   \[ W = W - \frac{\eta}{\sqrt{v_t} + \epsilon} m_t \]

   - Adaptive learning rates, best for deep CNNs.

3. **RMSprop**

   - Helps handle varying gradient scales.

4. **Nadam (Nesterov-accelerated Adam)**
   - Enhances Adam with **Nesterov momentum**.

---

## **Dropout & Regularization**

Regularization prevents **overfitting**.

### **1. Dropout**

- Deactivates random neurons to improve generalization.
- Common values: **0.2 - 0.5**.

### **2. L1 & L2 Regularization**

- **L1 (Lasso)**: Promotes feature selection.
  \[ L_1 = \lambda \sum |W| \]

- **L2 (Ridge, Weight Decay)**: Reduces large weights.
  \[ L_2 = \lambda \sum W^2 \]

- Common **λ** value: **0.0001**.

---

## **Model Training Approaches**

1. **Basic CNN Model**

   - Uses **Conv layers → ReLU → Pooling → Dense layers → Softmax**.

2. **Pretrained Models (Transfer Learning) [Recommended]**

   - Use pre-trained networks like **ResNet-18, VGG16, EfficientNet**.

3. **Fine-Tuning**
   - **Unfreeze top layers** of a pretrained model and train further.

---

## **Evaluation Metrics**

1. **Accuracy**:
   \[ \text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Samples}} \times 100 \]

2. **Precision & Recall**:

   - **Precision**: Measures correct positive predictions.
   - **Recall**: Measures how many actual positives were predicted.

3. **F1-Score**:
   \[ F1 = \frac{2 \times (\text{Precision} \times \text{Recall})}{\text{Precision} + \text{Recall}} \]
