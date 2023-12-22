import os
import pickle
from qclass import MyClass
from qibo import set_backend
from help_functions import plot_metrics

set_backend("tensorflow")


def main():
    nome_file = "epochs.txt"
    epochs = 2
    learning_rate = 0.01
    training_sample = 26
    method = "Adam"
    batch_size = 2
    layers = 2

    my_class = MyClass(
        epochs=epochs,
        learning_rate=learning_rate,
        training_sample=training_sample,
        method=method,
        batch_size=batch_size,
        nome_file=nome_file,
        nome_barplot=nome_barplot,
        name_predictions=name_predictions,
        layers=layers[j],
        seed_value=seed,
        test_sample=test_sample,
        nqubits=nqubits[k],
        resize=resize,
        block_width=block_sizes[k][0],
        block_heigth=block_sizes[k][1],
    )
