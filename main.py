import os
import pickle
from qclass import MyClass
from qibo import set_backend
from help_functions import plot_metrics

set_backend("tensorflow")


def main():
    nome_file = "epochs.txt"
    epochs = 2
    learning_rate = 0.1
    training_sample = 100
    method = "Adam"
    batch_size = 5
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

    # Printa initial parameters
    vparams = my_class.get_parameters()
    with open(nome_file, "a") as file:
        print(f"Inizio allenamento. Parametri iniziali:\n{vparams[0:20]}", file=file)
        print("=" * 60, file=file)

    # Initialize data
    my_class.initialize_data()

    # Barplot
    my_class.barplot()

    # Training
    epoch_train_loss, epoch_validation_loss, params, epochs = my_class.training_loop()

    # Plot training and validation loss
    plot_metrics(epochs, epoch_train_loss, method, epoch_validation_loss)

    # Test loop
    accuracy = my_class.test_loop()

    # Save final parameters
    with open("saved_parameters.pkl", "wb") as f:
        pickle.dump(params, f, pickle.HIGHEST_PROTOCOL)

    # Print Accuracy Test set
    with open(nome_file, "a") as file:
        print("/" * 60, file=file)
        print("/" * 60, file=file)
        print(f"Accuracy test set: {accuracy.result().numpy()}", file=file)
        print("/" * 60, file=file)
        print("/" * 60, file=file)


if __name__ == "__main__":
    main()
