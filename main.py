import os
import pickle
import numpy as np
import seaborn as sns
from qclass import MyClass
from qibo import set_backend
from help_functions import plot_metrics, heatmap, histo

set_backend("tensorflow")


def main():
    epochs = 100
    learning_rate = 0.05
    training_sample = 500
    test_sample = 100
    method = "Adam"
    batch_size = 30
    layers = [1, 2, 3, 4, 5, 6]
    seed = 0
    block_sizes = [[2, 4], [3, 4], [4, 4], [4, 8]]
    nqubits = [8, 6, 4, 2]
    resize = 8

    nome_barplot = "barplot.png"
    accuracy = []
    for k in range(len(nqubits)):
        for j in range(len(layers)):
            # Nome files
            nome_file = f"history_q{nqubits[k]}_l{layers[j]}.txt"
            name_metrics = f"loss_q{nqubits[k]}_l{layers[j]}.png"
            name_params = f"params_q{nqubits[k]}_l{layers[j]}.pkl"
            name_predictions = f"predictions_q{nqubits[k]}_l{layers[j]}_"

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
                test_sample=test_sample,
                nqubits=nqubits[k],
                resize=resize,
                block_width=block_sizes[k][0],
                block_heigth=block_sizes[k][1],
            )

            # Initialize data
            my_class.initialize_data()

            # Barplot to check balance of data
            my_class.barplot()

            # Test loop before training
            name = f"q{nqubits[k]}_l{layers[j]}_before"
            acc, predictions, labels = my_class.test_loop("before")
            accuracy.append(acc.result().numpy())
            histo(predictions, labels, acc.result().numpy(), name)
            with open(nome_file, "a") as file:
                print("/" * 60, file=file)
                print("/" * 60, file=file)
                print(f"Layer = {layers[j]}", file=file)
                print(
                    f"Accuracy test set (before): {acc.result().numpy()}",
                    file=file,
                )
                print("/" * 60, file=file)
                print("/" * 60, file=file)

            # Training loop
            (
                epoch_train_loss,
                epoch_train_accuracy,
                epoch_validation_loss,
                params,
                epochs,
            ) = my_class.training_loop()

            # Plot training and validation loss
            plot_metrics(
                epochs,
                epoch_train_loss,
                epoch_train_accuracy,
                method,
                name_metrics,
                epoch_validation_loss,
            )

            # Test loop after training
            name = f"q{nqubits[k]}_l{layers[j]}_after"
            acc, predictions, labels = my_class.test_loop("after")
            accuracy.append(acc.result().numpy())
            histo(predictions, labels, acc.result().numpy(), name)

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

    # Final summary of accuracies
    with open("summary.txt", "a") as file:
        for k in range(len(nqubits)):
            for i in range(len(layers)):
                print("/" * 60, file=file)
                print("/" * 60, file=file)
                print(f"Number of qubits: {nqubits[k]}", file=file)
                print(
                    f"(Layers, Accuracy) = ({layers[i], accuracy[i + len(layers)*k]})",
                    file=file,
                )
                print("/" * 60, file=file)
                print("/" * 60, file=file)

    # Heatmap
    heatmap(accuracy, nqubits, layers)


if __name__ == "__main__":
    main()
