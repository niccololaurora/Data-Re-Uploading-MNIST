import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ================
# Batch functions
# ================


def batch_data(x_train, y_train, number_of_batches, sizes_batches):
    x_batch = []
    y_batch = []

    for k in range(number_of_batches):
        if k == number_of_batches - 1:
            x = x_train[
                sizes_batches[k - 1] * k : sizes_batches[k - 1] * k + sizes_batches[k]
            ]
            y = y_train[
                sizes_batches[k - 1] * k : sizes_batches[k - 1] * k + sizes_batches[k]
            ]
            x_batch.append(x)
            y_batch.append(y)
        else:
            x = x_train[sizes_batches[k] * k : sizes_batches[k] * (k + 1)]
            y = y_train[sizes_batches[k] * k : sizes_batches[k] * (k + 1)]
            x_batch.append(x)
            y_batch.append(y)

    return x_batch, y_batch


def calculate_batches(x_train, batch_size):
    if len(x_train) % batch_size == 0:
        number_of_batches = int(len(x_train) / batch_size)
        sizes_batches = [batch_size for i in range(number_of_batches)]
    else:
        number_of_batches = int(len(x_train) / batch_size) + 1
        size_last_batch = len(x_train) - batch_size * int(len(x_train) / batch_size)
        sizes_batches = [batch_size for i in range(number_of_batches - 1)]
        sizes_batches.append(size_last_batch)

    return number_of_batches, sizes_batches


# ================
# Plot functions
# ================


def plot_metrics(
    nepochs,
    train_loss_history,
    train_accuracy_history,
    method,
    name_metrics,
    validation_loss_history=None,
):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    epochs = np.arange(0, nepochs, 1)
    ax[0].plot(epochs, train_loss_history, label="Training Loss")
    ax[1].plot(epochs, train_accuracy_history, label="Training Accuracy")

    if validation_loss_history is not None:
        ax[0].plot(epochs, validation_loss_history, label="Validation Loss")

    ax[0].set_title("Loss")
    ax[1].set_title("Accuracy")
    ax[0].legend()
    ax[1].legend()
    ax[0].set_xlabel("Epochs")
    ax[1].set_xlabel("Epochs")
    plt.savefig(name_metrics)
    plt.close()


def plot_predictions(predictions, x_data, y_data, name):
    rows = 2
    columns = 2
    fig, ax = plt.subplots(nrows=rows, ncols=columns, figsize=(12, 12))

    for i in range(rows):
        for j in range(columns):
            rounded_prediction = round(predictions[i * rows + j], 2)
            ax[i][j].imshow(x_data[i * rows + j], cmap="gray")

            is_correct = (
                predictions[i * rows + j] >= 0.5 and y_data[i * rows + j] == 1
            ) or (predictions[i * rows + j] < 0.5 and y_data[i * rows + j] == 0)
            title_color = "green" if is_correct else "red"
            ax[i][j].set_title(f"Prediction: {rounded_prediction}", color=title_color)
            ax[i][j].set_xticks([])
            ax[i][j].set_yticks([])

    plt.savefig(name)
    plt.close()


def heatmap(accuracy, nqubits, layers):
    accuracy_matrix = np.array(accuracy).reshape(len(nqubits), len(layers))
    sns.heatmap(
        accuracy_matrix,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        xticklabels=layers,
        yticklabels=nqubits,
    )
    plt.xlabel("Number of Layers")
    plt.ylabel("Number of Qubits")
    plt.title("Accuracy Heatmap")
    plt.savefig("heatmap.png")
    plt.close()


def histo(predictions, labels, accuracy, name):
    # Costruisco due liste
    # La prima lista contiene le predizioni che il modello ha fatto quando l'immagine era uno zero
    zeros_predictions = [pred for pred, label in zip(predictions, labels) if label == 0]
    ones_predictions = [pred for pred, label in zip(predictions, labels) if label == 1]

    plt.hist(
        [zeros_predictions, ones_predictions],
        bins=20,
        color=["red", "blue"],
        label=["Zeros", "Ones"],
        histtype="step",
        linewidth=1.5,
    )

    plt.xlabel("Predictions")
    plt.ylabel("Frequency")
    plt.title("Predictions distribution")
    plt.legend()

    plt.text(
        0.5,
        0.9,
        f"Accuracy: {accuracy:.2%}",
        transform=plt.gca().transAxes,
        color="black",
        fontsize=10,
        ha="center",
    )

    name_file = "distribution_" + name + "_.png"
    plt.savefig(name_file)
    plt.close()
