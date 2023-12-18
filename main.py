import os
from qclass import MyClass
from qibo import set_backend
from help_functions import batch_data, calculate_batches

set_backend("tensorflow")


def main():
    epochs = 0
    learning_rate = 0
    training_sample = 0
    method = 0
    batch_size = 32

    my_class = MyClass(
        epochs=epochs,
        learning_rate=learning_rate,
        training_sample=training_sample,
        method=method,
    )

    vparams = my_class.get_parameters()

    with open("epochs.txt", "a") as file:
        print(f"Inizio allenamento. Parametri iniziali:\n{vparams[0:20]}", file=file)
        print("=" * 60, file=file)

    x_train, y_train = my_class.initialize_data()
    best = 0
    params = 0
    extra = 0

    number_of_batches, sizes_batches = calculate_batches(x_train, batch_size)
    # Serve uno shuffle dei dati prima di ogni nuova epoca?

    for i in range(epochs):
        for k in range(number_of_batches):
            x, y = batch_data(k, x_train, y_train, number_of_batches, batch_size)
            best, params, extra = my_class.training_loop(x, y)

        with open("epochs.txt", "a") as file:
            print("=" * 60, file=file)
            print(f"Epoch {i+1}", file=file)
            print(f"Parametri finali {params[0:20]}", file=file)
            print("=" * 60, file=file)

    my_class.plot_metrics()
    accuracy = my_class.test_loop()

    with open("epochs.txt", "a") as file:
        print("/" * 60, file=file)
        print("/" * 60, file=file)
        print(f"Accuracy {accuracy.result().numpy()}", file=file)
        print("/" * 60, file=file)
        print("/" * 60, file=file)


if __name__ == "__main__":
    main()
