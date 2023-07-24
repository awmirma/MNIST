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

