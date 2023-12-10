import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# Carica il dataset MNIST
(x_train, y_train), (_, _) = mnist.load_data()

mask_train = (y_train == 0) | (y_train == 1)
x_train = x_train[mask_train]
print("=" * 60)
print(x_train[0])

x_train = x_train[0:10]
print("=" * 60)
print(x_train[0])


# x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
# print("=" * 60)
# print(x_train[0])

x_train = tf.expand_dims(x_train, axis=-1)
print("=" * 60)
print(x_train[0])


x_train = tf.image.resize(x_train, [9, 9])
print("=" * 60)
print(x_train[0])
