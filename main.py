import os
from qclass import MyClass
from qibo import set_backend

set_backend("tensorflow")


def main():
    epochs = input("Insert epochs: ")
    learning_rate = input("Insert learning rate: ")
    training_sample = input("Insert size training sample: ")
    while True:
        method = input("Insert optimizer: ")
        if method.istitle():
            break
        else:
            print("The optimizer must start with a capital letter.")

    my_class = MyClass(
        epochs=epochs,
        learning_rate=learning_rate,
        training_sample=training_sample,
        method=method,
    )

    vparams = my_class.get_parameters()

    with open("file.txt", "a") as file:
        print(f"Parametri {vparams[0:20]}", file=file)
        print("=" * 60, file=file)

    my_class.initialize_data()
    best, params, extra = my_class.training_loop()

    with open("file.txt", "a") as file:
        print(f"Parametri finali {params[0:20]}", file=file)
        print("=" * 60, file=file)

    my_class.plot_metrics()
    accuracy = my_class.test_loop()

    with open("file.txt", "a") as file:
        print("/" * 60, file=file)
        print("/" * 60, file=file)
        print(f"Accuracy {accuracy.result().numpy()}", file=file)
        print("/" * 60, file=file)
        print("/" * 60, file=file)


if __name__ == "__main__":
    main()
