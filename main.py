import os
import pickle
import numpy as np
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
    layers = [1, 2, 3, 4, 5, 6, 7, 8]
    seed = 0

    nome_barplot = "barplot.png"
    accuracy = []
    for j in range(len(layers)):
        accuracy_layer = []
        for i in range(10):
            # Nome files
            nome_file = f"layer_{layers[j]}_" + f"rep_{i}" + "_.txt"
            name_metrics = f"loss_layer_{layers[j]}_" + f"rep_{i}" + "_.png"
            name_params = f"params_layer_{layers[j]}_" + f"rep_{i}" + "_.pkl"
            name_predictions = f"predictions_layer_{layers[j]}_" + f"rep_{i}_"

            with open(nome_file, "a") as file:
                print(f"Layer = {layers[j]}", file=file)
                print(f"Trial = {i}", file=file)

            # Update seed
            seed += 1

            # Create class
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
            )

            # Initialize data
            my_class.initialize_data()

            # Barplot to check balance of data
            my_class.barplot()

            # Test loop before training
            acc = my_class.test_loop("before")
            with open(nome_file, "a") as file:
                print("/" * 60, file=file)
                print("/" * 60, file=file)
                print(f"Accuracy test set (before): {acc.result().numpy()}", file=file)
                print("/" * 60, file=file)
                print("/" * 60, file=file)

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
            accuracy_layer.append(acc.result().numpy())

            # Save final parameters
            with open(name_params, "wb") as f:
                pickle.dump(params, f, pickle.HIGHEST_PROTOCOL)

            # Print Accuracy Test set
            with open(nome_file, "a") as file:
                print("/" * 60, file=file)
                print("/" * 60, file=file)
                print(f"Accuracy test set (after): {acc.result().numpy()}", file=file)
                print("/" * 60, file=file)
                print("/" * 60, file=file)

        # Calculate accuracy and deviation std
        acc = sum(accuracy_layer) / len(accuracy_layer)
        sigma_acc = np.std(accuracy_layer)
        dict_acc = {"Accuracy": acc, "Deviazione Standard": sigma_acc}
        accuracy.append(dict_acc)

    # Final summary of accuracies
    with open("summary.txt", "a") as file:
        for i, acc_dict in enumerate(accuracy):
            print("/" * 60, file=file)
            print("/" * 60, file=file)
            for key, value in acc_dict.items():
                print(f"Number of layers: {i}", file=file)
                print(f"{key}: {value}", file=file)


if __name__ == "__main__":
    main()
