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
    training_sample = 10
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
        layers=layers,
    )

    # Carico i parametri
    with open("saved_parameters.pkl", "rb") as file:
        vparams = pickle.load(file)

    my_class.set_parameters(vparams)

    # Initialize data
    my_class.initialize_data()

    # Test loop
    accuracy = my_class.test_loop()
