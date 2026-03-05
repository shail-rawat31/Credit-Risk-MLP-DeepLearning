# Multiclass Image Classification using CNN and Transfer Learning

## Deep Learning Architectures and Techniques – Laboratory Assignment

This project implements **multiclass image classification** using **Convolutional Neural Networks (CNNs)** and **transfer learning**.

Two models are developed and compared:

1. Custom CNN architecture
2. Transfer learning model using a pretrained CNN (MobileNetV2)

The objective is to understand convolution operations, pooling layers, feature extraction, and the benefits of transfer learning in computer vision tasks.

---

# Project Objective

Image classification is a fundamental task in computer vision. Convolutional Neural Networks are designed to automatically extract spatial features from images.

This project demonstrates how CNNs can be used to classify images into multiple categories and how pretrained networks improve model performance.

---

# Dataset

The project uses the **Fashion-MNIST dataset**.

### Dataset Properties

| Property | Value |
|--------|--------|
| Total Images | 70,000 |
| Training Images | 60,000 |
| Test Images | 10,000 |
| Image Size | 28 × 28 |
| Number of Classes | 10 |

### Classes

- T-shirt
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle Boot

The dataset is loaded directly using TensorFlow/Keras.

---

# Project Workflow

The project follows a deep learning pipeline:

1. Dataset loading
2. Image preprocessing
3. Image normalization
4. Train-validation-test split
5. Data augmentation
6. Custom CNN architecture design
7. Transfer learning implementation
8. Model training
9. Performance evaluation
10. Model comparison

---

# CNN Architecture

Custom CNN architecture:

Input Layer  
→ Conv2D (32 filters, ReLU)  
→ MaxPooling2D  
→ Conv2D (64 filters, ReLU)  
→ MaxPooling2D  
→ Flatten Layer  
→ Dense Layer (128 neurons)  
→ Dropout Layer  
→ Output Layer (Softmax)

### Activation Functions

ReLU – hidden layers  
Softmax – output layer

### Loss Function

Categorical Cross Entropy

### Optimizer

Adam

---

# Transfer Learning

Transfer learning is implemented using **MobileNetV2**, a pretrained convolutional neural network trained on ImageNet.

Steps:

1. Load MobileNetV2 without top layers
2. Freeze pretrained layers
3. Add custom dense layers
4. Train final classification layer

Transfer learning allows faster training and improved accuracy.

---

# Model Evaluation

Both models are evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

Training performance is visualized using:

- Accuracy curves
- Loss curves

---

# Tools and Technologies

The project uses:

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

---

# Project Structure


CNN-Image-Classification
│
├── CNN_Image_Classification.ipynb
├── cnn_classifier.py
├── README.md
└── CNN_Image_Classification_Report.pdf


---

# How to Run

1. Open Jupyter Notebook or Google Colab
2. Open:


CNN_Image_Classification.ipynb


3. Run all cells sequentially.

The notebook will train the models, generate plots, and evaluate performance.

---

# Learning Outcomes

This project demonstrates practical implementation of:

- Convolution operations
- Pooling layers
- Feature extraction
- Data augmentation
- Transfer learning
- CNN evaluation metrics

---

# Author

Shail Rawat
MCA – Deep Learning Architectures and Techniques

---

# License

This project is developed for academic purposes as part of laboratory coursework.