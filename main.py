from qclass import MyClass
from qibo import set_backend

set_backend("tensorflow")


def main():
    my_class = MyClass(resize=9)

    vparams = my_class.get_parameters()

    with open("file.txt", "a") as file:
        print(f"Parametri {vparams[0:20]}", file=file)
        print("=" * 60, file=file)

    my_class.initialize_data()
    best, params, extra = my_class.training_loop()

    with open("file.txt", "a") as file:
        print(f"Parametri finali {params[0:20]}", file=file)
        print("=" * 60, file=file)

    accuracy = my_class.test_loop()

    with open("file.txt", "a") as file:
        print("/" * 60, file=file)
        print("/" * 60, file=file)
        print(f"Accuracy {accuracy.result().numpy()}", file=file)
        print("/" * 60, file=file)
        print("/" * 60, file=file)


if __name__ == "__main__":
    main()
