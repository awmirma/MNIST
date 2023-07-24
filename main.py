# Import necessary libraries
import numpy as np
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import to_categorical


# Load and preprocess the data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the pixels values to range between 0 and 1
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# Reshape the image to a 1D array (flatten)
train_images = train_images.reshape((-1,28*28))
test_images  = test_images.reshape((-1,28*28))

# One-hot encode the lables
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Build the model architecture 
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(28 * 28,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

# compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Traing model
history = model.fit(train_images, train_labels, epochs=10, batch_size=128, validation_split=0.1)

# evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print("Test accuracy:", test_accuracy)


# markpredictions
predictions = model.predict(test_images[:10])
predicted_labels = np.argmax(predictions, axis=1)
print("Predicted labels:", predicted_labels)
