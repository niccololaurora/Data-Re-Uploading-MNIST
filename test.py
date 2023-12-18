import numpy as np
import tensorflow as tf
from qibo import Circuit, gates, hamiltonians, set_backend
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# Carica il dataset MNIST
epochs = np.arange(0, 10, 1)
print(epochs)


def entanglement_block():
    """
    Args: None
    Return: circuit with CZs
    """
    c = Circuit(9)
    for q in range(0, 8, 2):
        c.add(gates.CNOT(q, q + 1))
    for q in range(1, 7, 2):
        c.add(gates.CNOT(q, q + 1))
    c.add(gates.CNOT(8, 0))
    return c


c = entanglement_block()
print(c.draw())
