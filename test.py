import numpy as np
import tensorflow as tf
import seaborn as sns
from qibo.symbols import Z, I
from qibo import Circuit, gates, hamiltonians, set_backend
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from qclass import MyClass
from help_functions import plot_metrics, heatmap

"""
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
"""


def block_creator(image):
    blocks = []

    for i in range(0, image.shape[0], 2):
        for j in range(0, image.shape[1], 2):
            block = image[i : i + 2, j : j + 2]
            block = tf.reshape(block, (2, 2))
            blocks.append(block)

    return blocks


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = tf.expand_dims(x_train, axis=-1)
x_train = tf.image.resize(x_train, [9, 9])
x_train = x_train / 255.0


epochs = 100
learning_rate = 0.05
training_sample = 500  # 350
# validation_sample = 150
test_sample = 100
method = "Adam"
batch_size = 30
layers = [1, 2, 3, 4, 5, 6]
seed = 0
bloch_size = 2
nqubits = 4
resize = 8

nome_file = f"layer_1_.txt"
name_metrics = f"loss_layer_1_.png"
name_params = f"params_layer_1_.pkl"
name_predictions = f"predictions_layer1_"
nome_barplot = "barplot.png"

my_class = MyClass(
    epochs=epochs,
    learning_rate=learning_rate,
    training_sample=training_sample,
    method=method,
    batch_size=batch_size,
    nome_file=nome_file,
    nome_barplot=nome_barplot,
    name_predictions=name_predictions,
    layers=layers[1],
    seed_value=seed,
    test_sample=test_sample,
    nqubits=nqubits,
    bloch_size=bloch_size,
    resize=resize,
)

"""
n = [2, 3, 4]
ham = 0
for i in range(len(n)):
    for k in range(n[i]):
        ham = I(0) * Z(1)
    hamiltonian = hamiltonians.SymbolicHamiltonian(ham)

    c = Circuit(n[i])
    c.add(gates.H(0))
    c.add(gates.H(1))
    res = c()
    expectation_value = hamiltonian.expectation(res.state())

    print(f"Valore con n {n[i]}: {expectation_value}")

"""

accuracy = [1, 0.8, 0.9, 0.7, 0.5, 0.6]

# Numero di qubits e layers
nqubits = [2, 3]
nlayers = [1, 2, 3]

heatmap(accuracy, nqubits, nlayers)
