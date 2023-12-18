def ask_params():
    epochs = input("Inserisci il numero di epochs: ")
    learning_rate = input("Inserisci il learning rate: ")
    training_sample = input("Inserisci la dimensione del campione di addestramento: ")
    batch_size = input("Inserisci la batch size: ")

    while True:
        optimizer = input("Inserisci l'ottimizzatore: ")
        if optimizer.istitle():
            break
        else:
            print("L'ottimizzatore deve iniziare con una lettera maiuscola.")

    return epochs, learning_rate, training_sample, optimizer


def main():
    epochs, learning_rate, training_sample, batch_size, optimizer = ask_params()

    main_file = "main.py"

    # Leggi il contenuto del file
    with open(main_file, "r") as file:
        main_file_content = file.read()

    # Sostituisci i valori appropriati nel contenuto del file
    main_file_content = main_file_content.replace("epochs = 0", f"epochs = {epochs}")
    main_file_content = main_file_content.replace(
        "learning_rate = 0", f"learning_rate = {learning_rate}"
    )
    main_file_content = main_file_content.replace(
        "training_sample = 0", f"training_sample = {training_sample}"
    )
    main_file_content = main_file_content.replace(
        "method = 0", f"method = '{optimizer}'"
    )
    main_file_content = main_file_content.replace(
        "batch_size = 0", f"method = '{batch_size}'"
    )

    # Scrivi il nuovo contenuto nel file
    with open(main_file, "w") as file:
        file.write(main_file_content)

    print("Le informazioni sono state scritte nel file 'main.py'.")


if __name__ == "__main__":
    main()
