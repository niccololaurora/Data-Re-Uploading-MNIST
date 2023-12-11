import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from qibo.symbols import Z
from qibo import Circuit, gates, hamiltonians, set_backend
from qibo.optimizers import optimize

set_backend("tensorflow")


class MyClass:
    def __init__(
        self, resize, layers=1, training_sample=5, method="l-bfgs-b", binary="yes"
    ):
        self.epochs = 25
        self.learning_rate = 0.001
        self.train_size = training_sample
        self.x_train = 0
        self.y_train = 0
        self.x_test = 0
        self.y_test = 0
        self.block_size = 3
        self.filt = "yes"
        self.method = "sgd"
        self.binary = binary
        self.resize = resize
        self.loss_history = []
        self.vparams = np.random.normal(loc=0, scale=1, size=(198,))
        self.embed_params = np.random.normal(loc=0, scale=1, size=(162,))
        self.average_params = np.random.normal(loc=0, scale=1, size=(18,))
        self.max_params = np.random.normal(loc=0, scale=1, size=(18,))
        self.hamiltonian = hamiltonians.SymbolicHamiltonian(
            Z(0) * Z(1) * Z(2) * Z(3) * Z(4) * Z(5) * Z(6) * Z(7) * Z(8)
        )

    def get_parameters(self):
        return self.vparams

    def set_parameters(self, vparams):
        self.vparams = vparams

    def plot_metrics(self):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 5))

        epochs = np.arange(0, self.epochs, 1)
        ax.plot(epochs, self.loss_history)
        ax.set_title("Mnist")
        ax.set_xlabel("Epochs")

        plt.savefig("loss.png")

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
            x_test = x_test[self.train_size + 1 : (self.train_size + 1) * 2]
            y_test = y_test[self.train_size + 1 : (self.train_size + 1) * 2]

        # Resize images
        width, length = self.resize, self.resize
        # x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
        # x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)

        x_train = tf.expand_dims(x_train, axis=-1)
        x_test = tf.expand_dims(x_test, axis=-1)

        x_train = tf.image.resize(x_train, [width, length])
        x_test = tf.image.resize(x_test, [width, length])

        # Normalize pixel values to be between 0 and 1
        x_train = x_train / 255.0
        x_test = x_test / 255.0

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def average_block(self, simple_list):
        """
        Args: list of 9 values
        Return: circuit with embedded this 9 values
        """
        c = Circuit(9)
        for q, value in enumerate(simple_list):
            rx = self.vparams[q * 2 + 162] * value + self.vparams[(q * 2 + 1) + 162]
            c.add(gates.RX(q, theta=rx))
        return c

    def max_block(self, simple_list):
        """
        Args: list of 9 values
        Return: circuit with embedded this 9 values
        """
        c = Circuit(9)
        for q, value in enumerate(simple_list):
            rx = self.vparams[q * 2 + 180] * value + self.vparams[(q * 2 + 1) + 180]
            c.add(gates.RX(q, theta=rx))
        return c

    def entanglement_block(self):
        """
        Args: None
        Return: circuit with CZs
        """
        c = Circuit(9)
        for q in range(0, 8, 2):
            c.add(gates.CZ(q, q + 1))
        for q in range(1, 7, 2):
            c.add(gates.CZ(q, q + 1))
        c.add(gates.CZ(0, 8))
        return c

    def embedding_block(self, blocks):
        """
        Args: an image divided in blocks (9 blocks 3x3)
        Return: a qibo circuit with the embedded image
        """
        c = Circuit(9)
        for k, block in enumerate(blocks):
            for i, x in enumerate(block):
                ry_0 = self.vparams[i * 6] * x[0] + self.vparams[i * 6 + 1]
                rz_1 = self.vparams[i * 6 + 2] * x[1] + self.vparams[i * 6 + 3]
                ry_2 = self.vparams[i * 6 + 4] * x[2] + self.vparams[i * 6 + 5]
                c.add(gates.RY(k, theta=ry_0))
                c.add(gates.RZ(k, theta=rz_1))
                c.add(gates.RY(k, theta=ry_2))

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

        # EMBEDDING
        c_em = self.embedding_block(blocks)
        res_cem = c_em()

        # ENTANGLEMENT
        c_ent = self.entanglement_block()
        res_cent = c_ent(res_cem.state())

        # AVERAGE
        average_pooling_values = self.average_pooling(blocks)
        c_aver = self.average_block(average_pooling_values)
        res_aver = c_aver(res_cent.state())

        # ENTANGLEMENT
        res_cent = c_ent(res_aver.state())

        # MAX POOLING
        max_pooling_values = self.max_pooling(blocks)
        c_max = self.max_block(max_pooling_values)
        res_max = c_max(res_aver.state())

        # EXPECTATION
        expectation_value = self.hamiltonian.expectation(res_max.state())
        return expectation_value

    def loss_function(self, vparams=None):
        if vparams is None:
            vparams = self.vparams
        self.set_parameters(vparams)

        predictions = []
        for x in self.x_train:
            with open("file.txt", "a") as file:
                print(f"Immagine", file=file)
            """
            The outcome of the circuit will be a number in [-1, 1], hence
            lo traslo in [0, 1].
            """
            exp = self.circuit(x)
            output = (exp + 1) / 2
            predictions.append(output)

        cf = tf.keras.losses.BinaryCrossentropy()(self.y_train, predictions)
        self.loss_history.append(cf)
        return cf

    def test_loop(self):
        predictions = []
        for x in self.x_test:
            with open("file.txt", "a") as file:
                print(f"Immagine", file=file)
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

    def training_loop(self):
        if self.method == "sgd":
            # perform optimization
            options = {
                "optimizer": "Adam",
                "learning_rate": 0.001,
                "nepochs": self.epochs,
                "nmessage": 1,
            }
            best, params, extra = optimize(
                self.loss_function,
                self.vparams,
                method="sgd",
                options=options,
            )

        else:
            best, params, extra = optimize(
                self.loss_function,
                self.vparams,
                method="parallel_L-BFGS-B",
            )

        return best, params, extra
