


class my_class():

    def __init__(self, layers=1, training_sample=200, method="l-bfgs-b", binary="yes", resize):
        self.epochs = 100
        self.learning_rate = 0.001
        self.train_size = training_sample
        self.x_train = 0
        self.y_train = 0
        self.x_test = 0
        self.y_test = 0
        self.binary = binary
        self.resize = resize

    def initialize_data():
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        if train_size != 0:
            x_train = x_train[0:self.train_size]
            y_train = y_train[0:self.train_size]
            x_test = x_test[self.train_size + 1 : (self.train_size + 1) * 2]
            y_test = y_test[self.train_size + 1 : (self.train_size + 1) * 2]

        if filt == "yes":
            mask_train = (y_train == 0) | (y_train == 1)
            mask_test = (y_test == 0) | (y_test == 1)
            x_train = x_train[mask_train]
            y_train = y_train[mask_train]
            x_test = x_test[mask_test]
            y_test = y_test[mask_test]

        # Resize images
        width, length = self.resize, self.resize
        x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
        x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)

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

    def entanglement_block():
        c = Circuit(9)
        for q in range(0,8,2):
            c.add(gates.CZ(q, q+1))
        for q in range(1,7,2):
            c.add(gates.CZ(q, q+1))
        c.add(gates.CZ(0, 9))
        print(c.draw())

        return c


    def filter():
        c = Circuit(1)
        
        # x sono oggetti (3, 1)
        # block sarà del tipo = [[1,2,3], [4,5,6], [7,8,9]]
        for i, x in enumerate(block):
            ry_1 = self.params[i] * x[0] + self.params[i+1] 
            rz_1 = self.params[i+2] * x[1] + self.params[i+3] 
            ry_2 = self.params[i+4] * x[2] + self.params[i+5] 

            

            c.add(gates.RY(i, theta=ry_1))
            c.add(gates.RZ(i, theta=rz))
            c.add(gates.RZ(i, theta=ry_2))

            c.add(gates.RY(i, theta=ry_1))
            c.add(gates.RZ(i, theta=rz))
            c.add(gates.RZ(i, theta=ry_2))

            c.add(gates.RY(i, theta=ry_1))
            c.add(gates.RZ(i, theta=rz))
            c.add(gates.RZ(i, theta=ry_2))


        return c

    def loss_function(self, parameters=None):

        if parameters is None:
            parameters = self.params
        self.set_parameters(parameters)

        cf = 0
        for x, y in zip(self.x_train, self.y_train):
            cf += self.single_loss(x, y)
        return cf


    def training_loop():
        for i in range(epochs):
            with tf.GradientTape() as tape:
                l = loss_function()


# Data loading and filtering
x_train, y_train, x_test, y_test = initialize_data(train_size, resize, filt)


