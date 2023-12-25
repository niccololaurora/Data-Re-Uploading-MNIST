import numpy as np
import tensorflow as tf
import os
import qibo
import matplotlib.pyplot as plt
from qibo.symbols import Z, I
from qibo import Circuit, gates, hamiltonians, set_backend
from qibo.optimizers import optimize
from help_functions import (
    batch_data,
    calculate_batches,
    plot_predictions,
    states_visualization,
)

qibo.set_threads(1)
set_backend("tensorflow")


class MyClass:
    def __init__(
        self,
        epochs,
        learning_rate,
        training_sample,
        test_sample,
        batch_size,
        method,
        nome_file,
        seed_value,
        block_width,
        block_heigth,
        nqubits,
        layers,
        resize,
        nome_barplot,
        name_predictions,
        name_qsphere,
        output_folder,
        output_losses,
        output_predictions,
        output_distributions,
        output_qspheres,
        pooling,
    ):
        np.random.seed(seed_value)

        # TRAINING SPECIFICS
        self.train_size = training_sample
        self.test_size = test_sample
        self.epochs_early_stopping = epochs
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.method = method
        self.patience = 10
        self.tolerance = 1e-4
        self.validation_split = 0.3
        self.options = {
            "optimizer": self.method,
            "learning_rate": self.learning_rate,
            "nepochs": 1,
            "nmessage": 5,
        }

        # IMAGES SPECIFICS
        self.block_width = block_width
        self.block_heigth = block_heigth
        self.resize = resize
        self.filt = "yes"

        # DATASET
        self.x_train = 0
        self.y_train = 0
        self.batch_x = 0
        self.batch_y = 0
        self.x_test = 0
        self.y_test = 0
        self.x_validation = 0
        self.y_validation = 0

        # ARCHITECTURE
        self.nqubits = nqubits
        self.layers = layers
        self.n_embed_params = 2 * self.nqubits * self.block_width * self.block_heigth
        self.params_1layer = 2 * self.nqubits + self.n_embed_params
        # 180 = 2*9*9 + 2*9 per 9 qubit e blocchi 3x3
        # 2*self.nqubits*(1 + self.bloch_size**2)
        self.vparams = np.random.normal(
            loc=0, scale=1, size=(self.params_1layer * self.layers,)
        ).astype(np.complex128)
        self.pooling = pooling

        # PHYSICS
        self.final_state = 0
        self.hamiltonian = self.create_hamiltonian()

        # NOMI FILES
        self.nome_barplot = nome_barplot
        self.nome_file = nome_file
        self.name_predictions = name_predictions
        self.name_qsphere = name_qsphere
        self.output_folder = output_folder
        self.output_qspheres = output_qspheres
        self.output_distributions = output_distributions
        self.output_losses = output_losses
        self.output_predictions = output_predictions

    # ================
    # Miscellaneous functions
    # ================

    def create_hamiltonian(self):
        ham = 0
        for k in range(self.nqubits):
            ham = I(0) * Z(k)
        hamiltonian = hamiltonians.SymbolicHamiltonian(ham)
        return hamiltonian

    def get_parameters(self):
        return self.vparams

    def set_parameters(self, vparams):
        self.vparams = vparams

    # ================
    # Visual
    # ================

    def barplot(self):
        # Counting zeros and ones in each dataset
        train_zeros = np.sum(self.y_train == 0)
        train_ones = np.sum(self.y_train == 1)

        test_zeros = np.sum(self.y_test == 0)
        test_ones = np.sum(self.y_test == 1)

        validation_zeros = np.sum(self.y_validation == 0)
        validation_ones = np.sum(self.y_validation == 1)

        # Labels for the bars
        labels = ["Zeros", "Ones"]

        # Heights of the bars
        train_heights = [train_zeros, train_ones]
        test_heights = [test_zeros, test_ones]
        validation_heights = [validation_zeros, validation_ones]

        # Bar width
        bar_width = 0.35

        # Creating bar plots
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 5))
        ax[0].bar(
            np.arange(len(labels)), train_heights, bar_width, label="Training Set"
        )
        ax[1].bar(np.arange(len(labels)), test_heights, bar_width, label="Test Set")
        ax[2].bar(
            np.arange(len(labels)),
            validation_heights,
            bar_width,
            label="Validation Set",
        )

        # Adding labels and legend
        for axis, title in zip(ax, ["Training Set", "Test Set", "Validation Set"]):
            axis.set_xticks(np.arange(len(labels)))
            axis.set_xticklabels(labels)
            axis.set_ylabel("Number of Images")
            axis.set_title(title)

        plt.savefig(self.output_folder + self.nome_barplot)
        plt.close()

    def visualize_state_sequence(self, filename):
        image_files = [
            self.output_folder
            + self.output_qspheres
            + self.name_qsphere
            + "e"
            + str(epoch)
            + ".png"
            for epoch in range(self.epochs_early_stopping)
        ]
        images = [imageio.imread(file) for file in image_files]
        imageio.mimsave(self.output_folder + filename, images, duration=600)

    def histogram_separation(self, predictions, labels, accuracy, name):
        # Costruisco due liste
        # La prima lista contiene le predizioni che il modello ha fatto quando l'immagine era uno zero
        zeros_predictions = [
            pred for pred, label in zip(predictions, labels) if label == 0
        ]
        ones_predictions = [
            pred for pred, label in zip(predictions, labels) if label == 1
        ]

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

        name_file = (
            self.output_folder
            + self.output_distributions
            + "distribution_"
            + name
            + "_.png"
        )
        plt.savefig(name_file)
        plt.close()

    # ================
    # Circuit blocks
    # ================

    def average_block(self, simple_list, k):
        c = Circuit(self.nqubits)
        for q, value in enumerate(simple_list):
            # parametri layer precedenti + parametri embedding + q
            angle = (
                self.vparams[k * self.params_1layer + self.n_embed_params + 2 * q]
                * value
                + self.vparams[
                    k * self.params_1layer + self.n_embed_params + (2 * q + 1)
                ]
            )
            c.add(gates.RX(q, theta=angle))
        return c

    def max_block(self, simple_list, k):
        c = Circuit(self.nqubits)
        for q, value in enumerate(simple_list):
            # parametri layer precedenti + parametri embedding + q
            angle = (
                self.vparams[k * self.params_1layer + self.n_embed_params + 2 * q]
                * value
                + self.vparams[
                    k * self.params_1layer + self.n_embed_params + (2 * q + 1)
                ]
            )
            c.add(gates.RX(q, theta=angle))
        return c

    def entanglement_block(self):
        """
        Args: None
        Return: circuit with CZs
        """
        c = Circuit(self.nqubits)
        for q in range(0, self.nqubits - 1, 2):
            c.add(gates.CNOT(q, q + 1))
        for q in range(1, self.nqubits - 2, 2):
            c.add(gates.CNOT(q, q + 1))
        c.add(gates.CNOT(self.nqubits - 1, 0))
        return c

    def embedding_block(self, blocks, nlayer):
        c = Circuit(self.nqubits)
        for j, block in enumerate(blocks):
            for i, x in enumerate(block):
                # parametri layer precedenti + i
                angle = (
                    self.vparams[nlayer * self.params_1layer + i * 2] * x
                    + self.vparams[nlayer * self.params_1layer + (i * 2 + 1)]
                )

                if (
                    (i == 1)  # 02
                    or (i == 4)  # 35
                    or (i == 7)  # 68
                    or (i == 10)  # 911
                    or (i == 13)  # 1214
                    or (i == 16)  # 1517
                    or (i == 19)  # 1820
                    or (i == 22)  # 2123
                    or (i == 25)  # 2426
                    or (i == 28)  # 2729
                    or (i == 31)  # 3032
                    or (i == 34)  # 3334
                ):
                    c.add(gates.RZ(j, theta=angle))
                else:
                    c.add(gates.RY(j, theta=angle))

        return c

    def circuit(self, x):
        # Suddivido l'immagine in blocchi secondo la larghezza e la lunghezza richieste
        blocks = self.block_creator(x)

        # Pooling of each block
        average_pooling_values = self.average_pooling(blocks)
        max_pooling_values = self.max_pooling(blocks)

        # Entanglement block
        c_ent = self.entanglement_block()

        # Initial state
        tensor_size = 2**self.nqubits
        tensor_values = [1] + [0] * (tensor_size - 1)
        initial_state = tf.constant(tensor_values, dtype=tf.float32)

        # Layers loop
        for k in range(self.layers):
            # EMBEDDING
            c_em = self.embedding_block(blocks, k)
            res_cem = c_em(initial_state)

            # ENTANGLEMENT
            res_cent = c_ent(res_cem.state())

            # POOLING
            res_pooling = 0
            if self.pooling == "max":
                # MAX POOLING
                c_max = self.max_block(max_pooling_values, k)
                res_pooling = c_max(res_cent.state())
                initial_state = res_pooling.state()

            elif self.pooling == "average":
                # AVERAGE POOLING
                c_aver = self.average_block(average_pooling_values, k)
                res_pooling = c_aver(res_cent.state())
                initial_state = res_pooling.state()

            if k == self.layers - 1:
                # RECORD FINAL STATE FOR GIF
                self.final_state = res_pooling.state(numpy=True)
                break

            # ENTANGLEMENT
            res_cent = c_ent(res_pooling.state())
            initial_state = res_cent.state()

        # EXPECTATION
        final_state = initial_state
        expectation_value = self.hamiltonian.expectation(final_state)
        return expectation_value

    # ================
    # Image processing: max and average pooling,
    # block creator and data initilization
    # ================

    def initialize_data(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        if self.filt == "yes":
            mask_train = (y_train == 0) | (y_train == 1)
            mask_test = (y_test == 0) | (y_test == 1)
            x_train = x_train[mask_train]
            y_train = y_train[mask_train]
            x_test = x_test[mask_test]
            y_test = y_test[mask_test]

        if self.train_size != 0:
            x_train = x_train[0 : self.train_size]
            y_train = y_train[0 : self.train_size]
            validation_size = int(len(x_train) * self.validation_split)

            x_validation = x_train[:validation_size]
            y_validation = y_train[:validation_size]
            x_train = x_train[validation_size:]
            y_train = y_train[validation_size:]
            x_test = x_test[0 : self.test_size]
            y_test = y_test[0 : self.test_size]

        # Resize images
        width, length = self.resize, self.resize

        x_train = tf.expand_dims(x_train, axis=-1)
        x_test = tf.expand_dims(x_test, axis=-1)
        x_validation = tf.expand_dims(x_validation, axis=-1)

        x_train = tf.image.resize(x_train, [width, length])
        x_test = tf.image.resize(x_test, [width, length])
        x_validation = tf.image.resize(x_validation, [width, length])

        # Normalize pixel values to be between 0 and 1
        x_train = x_train / 255.0
        x_test = x_test / 255.0
        x_validation = x_validation / 255.0

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.x_validation = x_validation
        self.y_validation = y_validation

        # Batching
        number_of_batches, sizes_batches = calculate_batches(
            self.x_train, self.batch_size
        )
        self.batch_x, self.batch_y = batch_data(
            self.x_train,
            self.y_train,
            number_of_batches,
            sizes_batches,
        )

    def max_pooling(self, blocks):
        max_values = []
        for block in blocks:
            block = tf.reshape(block, [-1])
            max_values.append(max(block))

        return max_values

    def average_pooling(self, blocks):
        average_values = []
        for block in blocks:
            block = tf.reshape(block, [-1])
            mean = sum(block) / len(block)
            average_values.append(mean)

        return average_values

    def block_creator(self, image):
        blocks = []
        for i in range(0, image.shape[0], self.block_heigth):
            for j in range(0, image.shape[1], self.block_width):
                block = image[i : i + self.block_heigth, j : j + self.block_width]
                block = tf.reshape(block, [-1])
                blocks.append(block)
        return blocks

    # ================
    # Loss
    # ================

    def loss_function(self, vparams, batch_x, batch_y):
        if vparams is None:
            vparams = self.vparams
        self.set_parameters(vparams)

        predictions = []
        for x in batch_x:
            exp = self.circuit(x)
            output = (exp + 1) / 2
            predictions.append(output)

        cf = tf.keras.losses.BinaryCrossentropy()(batch_y, predictions)
        return cf

    # =================
    # Training, Validation, Test loops
    # =================

    def test_loop(self, correction_name=None):
        predictions = []
        for x in self.x_test:
            exp = self.circuit(x)
            output = (exp + 1) / 2
            predictions.append(output)

        accuracy = tf.keras.metrics.BinaryAccuracy(threshold=0.5)
        accuracy.update_state(self.y_test, predictions)

        if correction_name != None:
            name = (
                self.output_folder
                + self.output_predictions
                + self.name_predictions
                + f"_{correction_name}_.png"
            )
            plot_predictions(predictions, self.x_test, self.y_test, name)
            return accuracy, predictions, self.y_test
        return accuracy

    def validation_loop(self):
        predictions = []
        for x in self.x_validation:
            exp = self.circuit(x)
            output = (exp + 1) / 2
            predictions.append(output)

        cf = tf.keras.losses.BinaryCrossentropy()(self.y_validation, predictions)
        return cf

    def training_loop(self):
        if (
            (self.method == "Adadelta")
            or (self.method == "Adagrad")
            or (self.method == "Adam")
        ):
            best, params, extra = 0, 0, 0
            epoch_train_loss = []
            epoch_train_accuracy = []
            epoch_validation_loss = []
            for i in range(self.epochs):
                with open(self.nome_file, "a") as file:
                    print("=" * 60, file=file)
                    print(f"Epoch {i+1}", file=file)

                batch_train_loss = []
                for k in range(len(self.batch_x)):
                    best, params, extra = optimize(
                        self.loss_function,
                        self.vparams,
                        args=(self.batch_x[k], self.batch_y[k]),
                        method="sgd",
                        options=self.options,
                    )
                    batch_train_loss.append(best)

                # Loss training
                e_train_loss = sum(batch_train_loss) / len(batch_train_loss)
                epoch_train_loss.append(e_train_loss)

                # Accuracy training
                e_training_accuracy = self.test_loop()
                epoch_train_accuracy.append(e_training_accuracy.result().numpy())

                # Loss Validation
                validation_loss = self.validation_loop()
                epoch_validation_loss.append(validation_loss)

                # Figure for gif
                if self.nqubits < 3:
                    if i % 2 == 0:
                        states_visualization(
                            self.final_state,
                            self.output_folder
                            + self.output_qspheres
                            + self.name_qsphere,
                            i + 1,
                            True,
                        )
                else:
                    if i % 2 == 0:
                        states_visualization(
                            self.final_state,
                            self.output_folder
                            + self.output_qspheres
                            + self.name_qsphere,
                            i + 1,
                            False,
                        )

                with open(self.nome_file, "a") as file:
                    print(f"Loss training set: {e_train_loss}", file=file)
                    print(f"Loss validation set: {validation_loss}", file=file)

                # Early Stopping
                if self.early_stopping(epoch_train_loss, epoch_validation_loss) == True:
                    with open(self.nome_file, "a") as file:
                        print(">" * 60, file=file)
                        print(f"Early stopping at epoch {i + 1}.", file=file)
                        print(">" * 60, file=file)

                    self.epochs_early_stopping = i
                    break

        else:
            best, params, extra = optimize(
                self.loss_function,
                self.vparams,
                method="parallel_L-BFGS-B",
            )

        return (
            epoch_train_loss,
            epoch_train_accuracy,
            epoch_validation_loss,
            params,
            self.epochs_early_stopping,
        )

    def early_stopping(self, training_loss_history, validation_loss_history):
        best_validation_loss = np.inf
        epochs_without_improvement = 0

        for epoch in range(len(training_loss_history)):
            training_loss = training_loss_history[epoch]
            validation_loss = validation_loss_history[epoch]

            # Verifica se la loss di validazione ha migliorato
            if validation_loss < best_validation_loss - self.tolerance:
                best_validation_loss = validation_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= self.patience:
                self.epochs_early_stopping = epoch + 1
                with open(self.nome_file, "a") as file:
                    print(">" * 60, file=file)
                    print(
                        f"Epochs without improvement: {epochs_without_improvement}",
                        file=file,
                    )
                    print(">" * 60, file=file)

                return True

        with open(self.nome_file, "a") as file:
            print(
                f"Epochs without improvement: {epochs_without_improvement}",
                file=file,
            )
        return False
