import numpy as np
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import to_categorical

def load_mnist_data():
    # Load the MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Normalize the pixel values to a range between 0 and 1
    train_images = train_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0

    # Reshape the images to a 1D array (flatten)
    train_images = train_images.reshape((-1, 28 * 28))
    test_images = test_images.reshape((-1, 28 * 28))

    # One-hot encode the labels
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    return train_images, train_labels, test_images, test_labels

def build_model():
    # Build the model architecture
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(28 * 28,)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def train_model(model, train_images, train_labels, epochs=10, batch_size=128, validation_split=0.1):
    # Train the model
    history = model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

def evaluate_model(model, test_images, test_labels):
    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(test_images, test_labels)
    return test_accuracy

if __name__ == "__main__":
    train_images, train_labels, test_images, test_labels = load_mnist_data()
    model = build_model()
    train_model(model, train_images, train_labels)
    test_accuracy = evaluate_model(model, test_images, test_labels)
    print("Test accuracy:", test_accuracy)
