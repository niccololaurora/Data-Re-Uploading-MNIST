import os
import pickle
import numpy as np
import seaborn as sns
from qclass import MyClass
from qibo import set_backend
from help_functions import (
    plot_metrics,
    heatmap,
    histogram_separation,
    accuracy_vs_layers,
)

set_backend("tensorflow")


def main():
    # ==============
    # Configuration
    # ==============
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
    pooling = "max"

    # ==============
    # Files' name and creation folders
    # ==============
    nome_barplot = "barplot.png"
    output_folder = "visualization/"
    output_losses = "losses/"
    output_predictions = "predictions/"
    output_distributions = "distributions/"
    output_parameters = "trained_parameters/"
    output_qspheres = "qspheres/"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        os.makedirs(output_folder + output_losses)
        os.makedirs(output_folder + output_predictions)
        os.makedirs(output_folder + output_distributions)
        os.makedirs(output_folder + output_qspheres)

    if not os.path.exists(output_parameters):
        os.makedirs(output_parameters)

    # ==============
    # Loop
    # ==============
    accuracy = []
    for k in range(len(nqubits)):
        accuracy_qubits = []
        for j in range(len(layers)):
            # Nome files
            nome_file = f"history_q{nqubits[k]}_l{layers[j]}.txt"
            name_metrics = f"loss_q{nqubits[k]}_l{layers[j]}.png"
            name_params = f"params_q{nqubits[k]}_l{layers[j]}.pkl"
            name_predictions = f"predictions_q{nqubits[k]}_l{layers[j]}_"
            name_qsphere = f"qsphere_q{nqubits[k]}_l{layers[j]}_"

            # Create class
            my_class = MyClass(
                epochs=epochs,
                learning_rate=learning_rate,
                training_sample=training_sample,
                method=method,
                batch_size=batch_size,
                nome_file=nome_file,
                name_qsphere=name_qsphere,
                nome_barplot=nome_barplot,
                output_folder=output_folder,
                output_losses=output_losses,
                output_predictions=output_predictions,
                name_predictions=name_predictions,
                output_distributions=output_distributions,
                output_qspheres=output_qspheres,
                layers=layers[j],
                seed_value=seed,
                test_sample=test_sample,
                nqubits=nqubits[k],
                resize=resize,
                pooling=pooling,
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
            my_class.histogram_separation(
                predictions, labels, acc.result().numpy(), name
            )
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
                epochs_stop,
            ) = my_class.training_loop()

            # Plot training and validation loss
            plot_metrics(
                epochs_stop,
                epoch_train_loss,
                epoch_train_accuracy,
                method,
                output_folder + output_losses + name_metrics,
                epoch_validation_loss,
            )

            # Test loop after training
            name = f"q{nqubits[k]}_l{layers[j]}_after"
            acc, predictions, labels = my_class.test_loop("after")
            accuracy_qubits.append(acc.result().numpy())
            my_class.histogram_separation(
                predictions, labels, acc.result().numpy(), name
            )

            # Save final parameters
            with open(output_parameters + name_params, "wb") as f:
                pickle.dump(params, f, pickle.HIGHEST_PROTOCOL)

            # Print Accuracy Test set
            with open(nome_file, "a") as file:
                print("/" * 60, file=file)
                print("/" * 60, file=file)
                print(f"Accuracy test set (after): {acc.result().numpy()}", file=file)
                print("/" * 60, file=file)
                print("/" * 60, file=file)

        # Append accuracy_qubits list to the general accuracy
        accuracy.append(accuracy_qubits)

    # Final summary of accuracies
    with open("summary.txt", "a") as file:
        for k in range(len(nqubits)):
            print("/" * 60, file=file)
            print(f"Number of qubits: {nqubits[k]}", file=file)
            for i in range(len(layers)):
                print(
                    f"(Layers, Accuracy) = ({layers[i], accuracy[k][i]})",
                    file=file,
                )

    # Heatmap
    heatmap(accuracy, nqubits, layers)

    # Accuracy versus layers for each circuit
    accuracy_vs_layers(accuracy, nqubits, layers)


if __name__ == "__main__":
    main()
