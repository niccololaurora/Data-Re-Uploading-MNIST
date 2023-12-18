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


def plot_metrics(nepochs, loss_history):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 5))

    epochs = np.arange(0, nepochs, 1)
    ax.plot(epochs, loss_history)
    ax.set_title("Mnist")
    ax.set_xlabel("Epochs")
    plt.savefig("loss.png")
