import os
import pickle
from qclass import MyClass
from qibo import set_backend
from help_functions import plot_metrics

set_backend("tensorflow")


def main():
    epochs = 30
    learning_rate = 0.01
    training_sample = 200
    method = "Adam"
    batch_size = 20
    layers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    seed = 0

    for layer in layers:
        accuracy = []
        for i in range(10):
            nome_file = f"layer_{layer}_" + f"rep_{i}" + "_.txt"
            nome_barplot = f"barplot_layer_{layer}_" + f"rep_{i}" + "_.png"
            name_metrics = f"loss_layer_{layer}_" + f"rep_{i}" + "_.png"
            name_params = f"params_layer_{layer}_" + f"rep_{i}" + "_.pkl"
            name_predictions = f"predictions"
            seed += 1
            my_class = MyClass(
                epochs=epochs,
                learning_rate=learning_rate,
                training_sample=training_sample,
                method=method,
                batch_size=batch_size,
                nome_file=nome_file,
                nome_barplot=nome_barplot,
                name_loss=name_metrics,
                layers=layers,
                seed_value=seed,
            )

            # Initialize data
            my_class.initialize_data()

            # Barplot to check balance of data
            my_class.barplot()

            # Test loop before training
            acc = my_class.test_loop("before")

            # Training loop
            (
                epoch_train_loss,
                epoch_validation_loss,
                params,
                epochs,
            ) = my_class.training_loop()

            # Plot training and validation loss
            plot_metrics(
                epochs, epoch_train_loss, method, name_metrics, epoch_validation_loss
            )

            # Test loop after training
            acc = my_class.test_loop("after")
            accuracy.append(acc)

            # Save final parameters
            with open(name_params, "wb") as f:
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
