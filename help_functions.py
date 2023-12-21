import numpy as np
import matplotlib.pyplot as plt


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


def plot_metrics(
    nepochs, train_loss_history, method, name_metrics, validation_loss_history=None
):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 5))

    epochs = np.arange(0, nepochs, 1)
    ax.plot(epochs, train_loss_history, label="Train Loss")

    if validation_loss_history is not None:
        ax.plot(epochs, validation_loss_history, label="Validation Loss")

    ax.set_title(method)
    ax.legend()
    ax.set_xlabel("Epochs")
    plt.savefig(name_metrics)


def plot_predictions(rows, columns, predictions, x_data, y_data, name):
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
