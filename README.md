# Neural Network–Based Credit Risk Prediction using Multi-Layer Perceptron

## Deep Learning Architectures and Techniques – Lab Assignment

This project implements a **Multi-Layer Perceptron (MLP) neural network** to perform **credit risk prediction** using structured financial data.

The goal of the model is to classify loan applicants as **good credit risk** or **bad credit risk** based on multiple financial attributes.

The implementation demonstrates fundamental deep learning concepts including:

- Artificial neurons
- Feedforward neural networks
- Activation functions
- Loss functions
- Optimization algorithms

---

# Project Objective

Financial institutions need to evaluate the creditworthiness of loan applicants to reduce financial risk. Machine learning models can assist in automating this process by learning patterns from historical financial data.

This project builds and trains a **feedforward neural network** that predicts credit risk using tabular financial data.

---

# Dataset

The project uses the **German Credit Dataset**, a widely used dataset for credit risk classification problems.

### Dataset Characteristics

| Feature | Description |
|-------|-------------|
| Number of records | ~1000 |
| Feature types | Numerical and categorical |
| Target variable | Credit Risk |

### Target Classes

| Class | Meaning |
|------|--------|
| 1 | Good Credit Risk |
| 0 | Bad Credit Risk |

---

# Project Workflow

The project follows a typical deep learning pipeline:

1. Data loading
2. Data preprocessing
3. Handling missing values
4. Encoding categorical variables
5. Feature scaling
6. Train–test split
7. MLP architecture design
8. Model training
9. Visualization of training results
10. Performance evaluation

---

# Neural Network Architecture

The model uses a **feedforward neural network architecture** consisting of:

Input Layer  
→ Dense Layer (32 neurons, ReLU)  
→ Dense Layer (16 neurons, ReLU)  
→ Output Layer (1 neuron, Sigmoid)

### Activation Functions

- **ReLU** for hidden layers
- **Sigmoid** for output layer

### Loss Function

Binary Cross Entropy

### Optimizer

Adam Optimizer

---

# Model Evaluation

The trained model is evaluated using several classification metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

These metrics help measure the effectiveness of the credit risk prediction model.

---

# Tools and Technologies

The project is implemented using the following technologies:

- Python
- TensorFlow / Keras
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

---

# Project Structure
Credit-Risk-MLP
│
├── Credit_Risk_MLP.ipynb
├── credit_risk_mlp.py
├── README.md
└── Credit_Risk_MLP_Report.pdf


---

# How to Run the Project

## Using Jupyter Notebook

1. Open Jupyter Notebook or Google Colab
2. Open the notebook