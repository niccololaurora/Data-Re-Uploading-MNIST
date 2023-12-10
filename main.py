from qclass import MyClass


def main():
    my_class = MyClass(resize=9)
    my_class.initialize_data()
    best, params, extra = my_class.training_loop()


if __name__ == "__main__":
    main()
