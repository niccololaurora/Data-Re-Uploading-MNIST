def ask_params():
    epochs = input("Number of epochs: ")
    learning_rate = input("Learning rate: ")
    training_sample = input("Training size: ")
    batch_size = input("Batch size: ")
    layers = input("Number of layers: ")

    while True:
        optimizer = input("Optimizer: ")
        if optimizer.istitle():
            break
        else:
            print("The name of the optimizer must start with a capital letter.")

    return epochs, learning_rate, training_sample, optimizer, batch_size, layers


def main():
    (
        epochs,
        learning_rate,
        training_sample,
        optimizer,
        batch_size,
        layers,
    ) = ask_params()

    main_file = "main.py"

    # Leggi il contenuto del file
    with open(main_file, "r") as file:
        main_file_content = file.read()

    # Sostituisci i valori appropriati nel contenuto del file
    main_file_content = main_file_content.replace(
        "method = 0", f"method = '{optimizer}'"
    )
    main_file_content = main_file_content.replace("epochs = 0", f"epochs = {epochs}")
    main_file_content = main_file_content.replace(
        "learning_rate = 0", f"learning_rate = {learning_rate}"
    )
    main_file_content = main_file_content.replace(
        "training_sample = 0", f"training_sample = {training_sample}"
    )
    main_file_content = main_file_content.replace(
        "batch_size = 0", f"batch_size = {batch_size}"
    )
    main_file_content = main_file_content.replace("layers = 0", f"layers = {layers}")

    # Scrivi il nuovo contenuto nel file
    with open(main_file, "w") as file:
        file.write(main_file_content)

    print("Informations written in 'main.py'.")


if __name__ == "__main__":
    main()
