import numpy as np
import tensorflow as tf
from qibo import Circuit, gates, hamiltonians, set_backend
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt


train_size = 10
validation_split = 0.2
test_size = 10

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


mask_train = (y_train == 0) | (y_train == 1)
mask_test = (y_test == 0) | (y_test == 1)
x_train = x_train[mask_train]
y_train = y_train[mask_train]
x_test = x_test[mask_test]
y_test = y_test[mask_test]


x_train = x_train[0:train_size]
y_train = y_train[0:train_size]
validation_size = int(len(x_train) * validation_split)

x_validation = x_train[:validation_size]
y_validation = y_train[:validation_size]
x_train = x_train[validation_size:]
y_train = y_train[validation_size:]
x_test = x_test[0:test_size]
y_test = y_test[0:test_size]

# Resize images
width, length = 9, 9

x_train = tf.expand_dims(x_train, axis=-1)
x_test = tf.expand_dims(x_test, axis=-1)
x_validation = tf.expand_dims(x_validation, axis=-1)

x_train = tf.image.resize(x_train, [width, length])
x_test = tf.image.resize(x_test, [width, length])
x_validation = tf.image.resize(x_validation, [width, length])

# Normalize pixel values to be between 0 and 1
x_train = x_train / 255.0
x_test = x_test / 255.0
x_validation = x_validation / 255.0


fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(12, 5))

for i in range(4):
    for j in range(2):
        ax[j][i].imshow(x_train[j + i])

plt.savefig("fig.png")
