import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from qibo.symbols import Z
from qibo import Circuit, gates, hamiltonians, set_backend
from qibo.optimizers import optimize
from help_functions import batch_data, calculate_batches


set_backend("tensorflow")

# embedding 162
# Pooling 18


class MyClass:
    def __init__(
        self,
        epochs,
        learning_rate,
        training_sample,
        method,
        batch_size,
        layers=1,
        resize=9,
    ):
        self.epochs = epochs
        self.method = method
        self.patience = (
            3  # Numero massimo di epoche senza miglioramenti sulla loss di validazione
        )
        self.tolerance = 1e-4
        self.learning_rate = learning_rate
        self.train_size = training_sample
        self.test_size = training_sample
        self.validation_size = int(training_sample / 2)
        self.x_train = 0
        self.y_train = 0
        self.x_test = 0
        self.y_test = 0
        self.x_validation = 0
        self.y_validation = 0
        self.batch_x = 0
        self.batch_y = 0
        self.block_size = 3
        self.batch_size = batch_size
        self.filt = "yes"
        self.method = method
        self.resize = resize
        self.layers = layers
        self.nqubits = 9
        self.params_1layer = 180
        self.vparams = np.random.normal(
            loc=0, scale=1, size=(self.params_1layer * self.layers,)
        ).astype(np.complex128)
        # self.embed_params = np.random.normal(loc=0, scale=1, size=(162,))
        # self.average_params = np.random.normal(loc=0, scale=1, size=(18,))
        # self.max_params = np.random.normal(loc=0, scale=1, size=(18,))
        self.hamiltonian = hamiltonians.SymbolicHamiltonian(
            Z(0) * Z(1) * Z(2) * Z(3) * Z(4) * Z(5) * Z(6) * Z(7) * Z(8)
        )
        self.options = {
            "optimizer": self.method,
            "learning_rate": self.learning_rate,
            "nepochs": 1,
            "nmessage": 5,
        }

    def get_parameters(self):
        return self.vparams

    def set_parameters(self, vparams):
        self.vparams = vparams

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
            x_test = x_test[0 : self.test_size]
            y_test = y_test[0 : self.test_size]
            x_validation = x_train[
                self.train_size + 1 : (self.train_size + 1) + self.validation_size
            ]
            y_validation = y_train[
                self.train_size + 1 : (self.train_size + 1) + self.validation_size
            ]

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

    def average_block(self, simple_list, k):
        """
        Args: list of 9 values
        Return: circuit with embedded this 9 values
        """
        c = Circuit(9)
        for q, value in enumerate(simple_list):
            rx = (
                self.vparams[(q * 2) + k * self.params_1layer] * value
                + self.vparams[(q * 2 + 1) + k * self.params_1layer]
            )
            c.add(gates.RX(q, theta=rx))
        return c

    def max_block(self, simple_list, k):
        """
        Args: list of 9 values
        Return: circuit with embedded this 9 values
        """
        c = Circuit(9)
        for q, value in enumerate(simple_list):
            rx = (
                self.vparams[(q * 2) + k * self.params_1layer] * value
                + self.vparams[(q * 2 + 1) + k * self.params_1layer]
            )
            c.add(gates.RX(q, theta=rx))
        return c

    def entanglement_block(self):
        """
        Args: None
        Return: circuit with CZs
        """
        c = Circuit(9)
        for q in range(0, 8, 2):
            c.add(gates.CNOT(q, q + 1))
        for q in range(1, 7, 2):
            c.add(gates.CNOT(q, q + 1))
        c.add(gates.CNOT(8, 0))
        return c

    def embedding_block(self, blocks, k):
        """
        Args: an image divided in blocks (9 blocks 3x3)
        Return: a qibo circuit with the embedded image
        """
        c = Circuit(9)
        for j, block in enumerate(blocks):
            for i, x in enumerate(block):
                ry_0 = (
                    self.vparams[(i * 6) + k * self.params_1layer] * x[0]
                    + self.vparams[(i * 6 + 1) + k * self.params_1layer]
                )
                rz_1 = (
                    self.vparams[(i * 6 + 2) + k * self.params_1layer] * x[1]
                    + self.vparams[(i * 6 + 3) + k * self.params_1layer]
                )
                ry_2 = (
                    self.vparams[(i * 6 + 4) + k * self.params_1layer] * x[2]
                    + self.vparams[(i * 6 + 5) + k * self.params_1layer]
                )
                c.add(gates.RY(j, theta=ry_0))
                c.add(gates.RZ(j, theta=rz_1))
                c.add(gates.RY(j, theta=ry_2))

        return c

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
        """
        Args: an image
        Return: una lista che contiene i riquadri (in versione flat) in cui
        è stata divisa l'immagine.

        Example:
        image = [[
            [0, 0, 0, 0, 3, 2, 4, 5],
            [0, 0, 0, 0, 1, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8],
            [0, 0, 0, 0, 3, 2, 4, 5],
            [0, 0, 0, 0, 1, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8],
        ]]
        blocks = [[0,0,0,0], [0,0,0,0], [3,2,1,6], [4,5,7,8], ..., [7,8,7,8]]
        """
        blocks = []

        for i in range(0, image.shape[0], self.block_size):
            for j in range(0, image.shape[1], self.block_size):
                # Extract the block
                block = image[i : i + self.block_size, j : j + self.block_size]
                block = tf.reshape(block, (3, 3))
                blocks.append(block)

        return blocks

    def circuit(self, x):
        # Suddivido l'immagine 9x9 in 9 blocchi 3x3 (appiattiti)
        blocks = self.block_creator(x)

        # Average pooling of each block
        average_pooling_values = self.average_pooling(blocks)

        # Entanglement block
        c_ent = self.entanglement_block()

        # Initial state
        tensor_size = 2**self.nqubits
        tensor_values = [1] + [0] * (tensor_size - 1)
        initial_state = tf.constant(tensor_values, dtype=tf.float32)
        for k in range(self.layers):
            # EMBEDDING
            c_em = self.embedding_block(blocks, k)
            res_cem = c_em(initial_state)

            # ENTANGLEMENT
            res_cent = c_ent(res_cem.state())

            # AVERAGE POOLING
            c_aver = self.average_block(average_pooling_values, k)
            res_aver = c_aver(res_cent.state())

            if k == self.layers - 1:
                break
            # ENTANGLEMENT
            res_cent = c_ent(res_aver.state())
            initial_state = res_cent.state()

        # EXPECTATION
        expectation_value = self.hamiltonian.expectation(res_aver.state())
        return expectation_value

    def early_stopping(self, training_loss_history, validation_loss_history):
        best_validation_loss = np.inf
        epochs_without_improvement = 0

        for epoch in range(len(training_loss_history)):
            training_loss = training_loss_history[epoch]
            validation_loss = validation_loss_history[epoch]

            # Verifica se la loss di validazione ha migliorato
            if validation_loss < best_validation_loss - tolerance:
                best_validation_loss = validation_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # Verifica se interrompere l'addestramento
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch + 1}.")
                return True

        return False

    def loss_function(self, vparams, batch_x, batch_y):
        if vparams is None:
            vparams = self.vparams
        self.set_parameters(vparams)

        predictions = []
        for x in batch_x:
            """
            The outcome of the circuit will be a number in [-1, 1], hence
            lo traslo in [0, 1].
            """
            exp = self.circuit(x)
            output = (exp + 1) / 2
            predictions.append(output)

        cf = tf.keras.losses.BinaryCrossentropy()(batch_y, predictions)
        return cf

    def test_loop(self):
        predictions = []
        for x in self.x_test:
            """
            The outcome of the circuit will be a number in [-1, 1], hence
            lo traslo in [0, 1].
            """
            exp = self.circuit(x)
            output = (exp + 1) / 2
            predictions.append(output)

        accuracy = tf.keras.metrics.BinaryAccuracy(threshold=0.5)
        accuracy.update_state(self.y_test, predictions)

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
            epoch_validation_loss = []
            early_stopping = []
            for i in range(self.epochs):
                with open("epochs.txt", "a") as file:
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

                    with open("epochs.txt", "a") as file:
                        print("/" * 60, file=file)
                        print(f"Batch {k+1}", file=file)
                        print(f"Parametri:\n{params[0:20]}", file=file)
                        print("/" * 60, file=file)

                e_train_loss = sum(batch_train_loss) / len(batch_train_loss)
                epoch_train_loss.append(e_loss)

                # Validation
                validation_loss = self.validation_loop()
                epoch_validation_loss.append(validation_loss)

                # Early Stopping
                if self.early_stopping(epoch_train_loss, epoch_validation_loss) == True:
                    with open("epochs.txt", "a") as file:
                        print("=" * 60, file=file)
                        print(f"Parametri finali:\n{params[0:20]}", file=file)
                        print("=" * 60, file=file)
                    break

        else:
            best, params, extra = optimize(
                self.loss_function,
                self.vparams,
                method="parallel_L-BFGS-B",
            )

        return epoch_loss, params, extra
