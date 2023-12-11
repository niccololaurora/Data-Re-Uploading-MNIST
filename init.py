def ask_params():
    epochs = input("Inserisci il numero di epochs: ")
    learning_rate = input("Inserisci il learning rate: ")
    training_sample = input("Inserisci la dimensione del campione di addestramento: ")

    while True:
        optimizer = input("Inserisci l'ottimizzatore: ")
        if optimizer.istitle():
            break
        else:
            print("L'ottimizzatore deve iniziare con una lettera maiuscola.")

    return epochs, learning_rate, training_sample, optimizer


def write_file(epochs, learning_rate, training_sample, optimizer):
    with open("parametri.py", "w") as file:
        file.write(f"epochs = {epochs}\n")
        file.write(f"learning_rate = {learning_rate}\n")
        file.write(f"training_sample = {training_sample}\n")
        file.write(f"optimizer = '{optimizer}'\n")


if __name__ == "__main__":
    epochs, learning_rate, training_sample, optimizer = ask_params()
    write_file(epochs, learning_rate, training_sample, optimizer)

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
    main_file_content = main_file_content.replace("method = 0", f"method = {optimizer}")


print("Le informazioni sono state scritte nel file 'parametri.py'.")
