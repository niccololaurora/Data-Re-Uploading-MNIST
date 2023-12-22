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
fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(12, 5))

for i in range(4):
    for j in range(2):
        ax[j][i].imshow(x_train[j + i])

plt.savefig("fig.png")
"""


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = tf.expand_dims(x_train, axis=-1)
x_train = tf.image.resize(x_train, [8, 8])
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
block_sizes = [[4, 2]]

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
    layers=1,
    seed_value=seed,
    test_sample=test_sample,
    nqubits=2,
    resize=resize,
    block_width=4,
    block_heigth=8,
)

blocco = my_class.block_creator(x_train[0])

print(x_train[0])
print("=" * 60)
print(blocco[0])
print("=" * 60)
print(blocco[1])
print("=" * 60)

c = my_class.embedding_block(blocco, 0)
print(c.draw())

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
