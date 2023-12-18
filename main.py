import os
import pickle
from qclass import MyClass
from qibo import set_backend
from help_functions import plot_metrics

set_backend("tensorflow")


def main():
    epochs = 0
    learning_rate = 0
    training_sample = 0
    method = 0
    batch_size = 0
    layers = 2

    my_class = MyClass(
        epochs=epochs,
        learning_rate=learning_rate,
        training_sample=training_sample,
        method=method,
        batch_size=batch_size,
        layers=layers,
    )

    vparams = my_class.get_parameters()

    with open("epochs.txt", "a") as file:
        print(f"Inizio allenamento. Parametri iniziali:\n{vparams[0:20]}", file=file)
        print("=" * 60, file=file)

    my_class.initialize_data()

    # Training
    epoch_loss, params, extra = my_class.training_loop()

    # Save final parameters
    with open("saved_parameters.pkl", "wb") as f:
        pickle.dump(params, f, pickle.HIGHEST_PROTOCOL)

    # Plot training loss
    plot_metrics(epochs, epoch_loss)

    # Test loop
    accuracy = my_class.test_loop()

    with open("epochs.txt", "a") as file:
        print("/" * 60, file=file)
        print("/" * 60, file=file)
        print(f"Accuracy {accuracy.result().numpy()}", file=file)
        print("/" * 60, file=file)
        print("/" * 60, file=file)


if __name__ == "__main__":
    main()
