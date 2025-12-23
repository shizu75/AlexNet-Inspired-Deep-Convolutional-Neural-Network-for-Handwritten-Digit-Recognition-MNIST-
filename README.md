# AlexNet-Inspired Deep Convolutional Neural Network for Handwritten Digit Recognition (MNIST)

## Overview
This project implements an **AlexNet-inspired deep convolutional neural network (CNN)** to perform handwritten digit classification on the **MNIST dataset**. Although MNIST is a grayscale, low-resolution dataset, this work deliberately adapts it to a **high-capacity CNN architecture** to study representation learning, scalability, and transfer-learning–style preprocessing.

The project emphasizes **architectural depth, data reshaping, training dynamics, and evaluation rigor**, making it suitable for **research portfolios, graduate applications, and deep learning coursework**.

---

## Problem Statement
Handwritten digit recognition is a canonical problem in computer vision. While lightweight models can achieve high accuracy on MNIST, this project explores:

- How **large-scale CNN architectures** behave on small datasets  
- The effect of **image upscaling and channel expansion**
- Training stability in **deep networks with millions of parameters**

---

## Dataset
- **Source:** MNIST (Keras built-in)
- **Training samples:** 60,000
- **Test samples:** 10,000
- **Original resolution:** 28×28 (grayscale)
- **Preprocessing:**  
  - Normalized to `[0,1]`  
  - Resized to **227×227**
  - Converted to **3-channel RGB** to match AlexNet-style input

---

## Methodology

### 1. Data Preprocessing
- Reshape grayscale images to `(28, 28, 1)`
- Normalize pixel values
- Resize images to `(227, 227)`
- Stack channels to simulate RGB input
- Custom batch generator used for memory efficiency

---

### 2. Model Architecture
The network follows an **AlexNet-like design**:

- **Convolutional Layers**
  - Large kernel (11×11) initial convolution
  - Progressive depth increase (96 → 256 → 384)
- **Pooling Layers**
  - MaxPooling for spatial downsampling
- **Fully Connected Layers**
  - Two 4096-unit dense layers
  - Dropout for regularization
- **Output Layer**
  - 10-class Softmax classifier

> The architecture intentionally exceeds MNIST’s complexity to demonstrate deep feature learning behavior.

---

### 3. Training Strategy
- **Optimizer:** Adam
- **Loss Function:** Sparse Categorical Cross-Entropy
- **Batch Size:** 32
- **Epochs:** Up to 5 (with Early Stopping)
- **Callback:** EarlyStopping on training accuracy

---

## Evaluation Metrics
- Classification accuracy
- Training and validation loss curves
- Confusion matrix visualization
- Probability-based predictions

---

## Results & Observations
- The model successfully converges despite over-parameterization
- Demonstrates robustness of deep CNNs on small datasets
- Highlights trade-offs between **model capacity and dataset size**
- Confusion matrix confirms correct digit discrimination patterns

---

## Visualizations
The project includes:
- Model architecture diagram
- Training & validation accuracy/loss curves
- Confusion matrix heatmap

These plots provide insight into **learning dynamics and class-wise performance**.

---

## Reproducibility
All experiments are fully reproducible using:
- TensorFlow / Keras
- NumPy
- OpenCV
- Matplotlib / Seaborn
- scikit-learn

The MNIST dataset is automatically downloaded via Keras.

---

## Research Significance
This work demonstrates:
- Practical adaptation of **large CNN architectures**
- Data engineering for deep learning compatibility
- Experimental analysis of **capacity vs. generalization**

It serves as a **foundation for transfer learning, architectural ablation studies, and CNN scalability research**.

---

## Future Work
- Replace MNIST with higher-resolution datasets (e.g., CIFAR-10, SVHN)
- Compare performance against lightweight CNNs
- Apply transfer learning using pretrained AlexNet-style weights
- Introduce batch normalization and learning rate scheduling

---

## Technologies Used
- Python
- TensorFlow / Keras
- NumPy
- OpenCV
- Matplotlib
- Seaborn
- scikit-learn

---

## Author Note
This project is intended for **research-oriented portfolios**, demonstrating deep learning architecture design, experimentation discipline, and evaluation rigor.
