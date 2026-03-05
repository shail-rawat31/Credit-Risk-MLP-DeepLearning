# -----------------------------------------------
# Fashion MNIST Image Classification using CNN
# -----------------------------------------------
# This program trains a Convolutional Neural Network (CNN)
# to classify images from the Fashion-MNIST dataset into
# 10 different clothing categories.

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import dataset and neural network tools from TensorFlow
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Import evaluation metrics
from sklearn.metrics import confusion_matrix, classification_report


# -----------------------------------------------
# Load Fashion MNIST Dataset
# -----------------------------------------------
# Dataset contains 28x28 grayscale images of clothing items
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()


# -----------------------------------------------
# Data Preprocessing
# -----------------------------------------------

# Normalize pixel values (0–255 → 0–1)
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape data to add channel dimension (required for CNN)
X_train = X_train.reshape(-1,28,28,1)
X_test = X_test.reshape(-1,28,28,1)

# Convert labels to categorical format (One-Hot Encoding)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# -----------------------------------------------
# Build CNN Model
# -----------------------------------------------
model = Sequential()

# First Convolution Layer
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
model.add(MaxPooling2D((2,2)))

# Second Convolution Layer
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2)))

# Flatten feature maps into a vector
model.add(Flatten())

# Fully Connected Layer
model.add(Dense(128,activation='relu'))

# Dropout to reduce overfitting
model.add(Dropout(0.5))

# Output Layer (10 classes)
model.add(Dense(10,activation='softmax'))


# -----------------------------------------------
# Compile Model
# -----------------------------------------------
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# -----------------------------------------------
# Train Model
# -----------------------------------------------
history = model.fit(
    X_train,
    y_train,
    epochs=10,
    validation_data=(X_test,y_test)
)


# -----------------------------------------------
# Plot Training vs Validation Accuracy
# -----------------------------------------------
plt.figure(figsize=(8,5))

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.title("Model Accuracy During Training")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.show()


# -----------------------------------------------
# Make Predictions
# -----------------------------------------------
pred = model.predict(X_test)

# Convert probabilities to class labels
pred = np.argmax(pred,axis=1)
true = np.argmax(y_test,axis=1)


# -----------------------------------------------
# Confusion Matrix
# -----------------------------------------------
cm = confusion_matrix(true,pred)

plt.figure(figsize=(8,6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Confusion Matrix for Fashion MNIST Classification")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.show()


# -----------------------------------------------
# Classification Report
# -----------------------------------------------
print("Classification Report:\n")
print(classification_report(true,pred))