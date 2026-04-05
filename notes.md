test model created as follows:

import tensorflow as tf
from tensorflow.keras import layers, models
import os

# 1. Load the MNIST dataset (handwritten digits)
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. Normalize the data
# Pixel values are 0-255; scaling them to 0.0-1.0 helps the network learn faster
x_train, x_test = x_train / 255.0, x_test / 255.0

# 3. Build a simple Sequential Neural Network
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)), # Flattens 28x28 image to a 784-length vector
    layers.Dense(128, activation='relu'),   # Hidden layer with 128 neurons
    layers.Dropout(0.2),                   # Randomly drops units to prevent overfitting
    layers.Dense(10, activation='softmax') # Output layer (10 digits, 0-9)
])

# 4. Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 5. Train the model
print("--- Training Model ---")
model.fit(x_train, y_train, epochs=5)

# 6. Evaluate the model performance
print("\n--- Evaluating Model ---")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\nTest accuracy: {test_acc:.4f}')

# 7. Export the model as a file
model_filename = 'mnist_simple_model.keras'
model.save(model_filename)
print(f"\nModel successfully saved as: {os.path.abspath(model_filename)}")
from google.colab import files
files.download(model_filename)

# 8. Export test dataset
import numpy as np
filename = 'mnist_test_data.npz'
np.savez_compressed(filename, x_test=x_test, y_test=y_test)
files.download(filename)