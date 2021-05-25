from keras.utils import np_utils
from keras.datasets import mnist
import numpy as np
from scipy import signal

import numpy as np


def linear(z, m):
    return m * z


def linear_prime(z, m):
    return m


def elu(z, alpha):
    return z if z >= 0 else alpha*((e**z) - 1)


def elu_prime(a, alpha):
    return 1 if z > 0 else alpha * np.exp(z)


def relu(z):
    return max(0, z)


def relu_prime(z):
    return 1 if z > 0 else 0


def leakyrelu(z, alpha):
    return max(alpha * z, z)


def leakyrelu_prime(z, alpha):
    return 1 if z > 0 else alpha


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# TODO softmax_prime


def tanh(x):
    return np.tanh(x)


def tanh_prime(x):
    return 1 - np.tanh(x)**2


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    return np.exp(-x) / (1 + np.exp(-x))**2


def relu(x):
    return np.maximum(x, 0)


def relu_prime(x):
    return np.array(x >= 0).astype('int')


# Loss Function
def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))


def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size


def sse(y_true, y_pred):
    return 0.5 * np.sum(np.power(y_true - y_pred, 2))


def sse_prime(y_true, y_pred):
    return y_pred - y_true


def CrossEntropy(yHat, y):
    if y == 1:
        return -log(yHat)
    else:
        return -log(1 - yHat)

# TODO CrossEntropy_prime function


class Layer():
    """
    base layer class
    """

    def __init__(self):
        self.input = None
        self.output = None

    def forward_propagation(self, input):
        """
        computes the output Y of a layer for a given input X
        """
        raise NotImplementedError

    def backward_propagation(self, output_error, learning_rate):
        """
        computes dE/dX for given dE/dY (and update parameters if any)
        """
        raise NotImplementedError


class FCLayer(Layer):
    """
    inherit from base class Layer
    """

    def __init__(self, input_size, output_size):
        """
        input_size = number of input neurons
        output_size = number of output neurons
        """
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

    def forward_propagation(self, input_data):
        """
        returns output for a giver input
        """
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        """
        computes dE/dW, dE/dB for a given output_error=dE/dY. returns input_error=dE/dX
        """
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)

        # update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error


class ActivationLayer(Layer):

    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward_propagation(self, input_data):
        """
        returns activated input
        """
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        """
        returns input_error=dE/dX for a given output_error=dE/dY
        learning rate is not used because there is no "learnable" parameter
        """
        return self.activation_prime(self.input) * output_error


class FlattenLayer:
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def forward(self, input):
        return np.reshape(input, (1, -1))

    def backward(self, output_error, learning_rate):
        return np.reshape(output_error, self.input_shape)


class ConvLayer(Layer):
    """
    inherit from base class Layer
    This convolutional layer is always with stride 1
    """

    def __init__(self, input_shape, kernel_shape, layer_depth):
        """
        input_shape = (i,j,d)
        kernel_shape = (m,n)
        layer_depth = output_depth
        """
        self.input_shape = input_shape
        self.input_depth = input_shape[2]
        self.kernel_shape = kernel_shape
        self.layer_depth = layer_depth
        self.output_shape = (
            input_shape[0]-kernel_shape[0]+1, input_shape[1]-kernel_shape[1]+1, layer_depth)
        self.weights = np.random.rand(
            kernel_shape[0], kernel_shape[1], self.input_depth, layer_depth) - 0.5
        self.bias = np.random.rand(layer_depth) - 0.5

    def forward_propagation(self, input):
        """ returns output for a given input """
        self.input = input
        self.output = np.zeros(self.output_shape)

        for k in range(self.layer_depth):
            for d in range(self.input_depth):
                self.output[:, :, k] += signal.correlate2d(
                    self.input[:, :, d], self.weights[:, :, d, k], 'valid') + self.bias[k]

        return self.output

    def backward_propagation(self, output_error, learning_rate):
        """computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX."""
        in_error = np.zeros(self.input_shape)
        dWeights = np.zeros(
            (self.kernel_shape[0], self.kernel_shape[1], self.input_depth, self.layer_depth))
        dBias = np.zeros(self.layer_depth)

        for k in range(self.layer_depth):
            for d in range(self.input_depth):
                in_error[:, :, d] += signal.convolve2d(
                    output_error[:, :, k], self.weights[:, :, d, k], 'full')
                dWeights[:, :, d, k] = signal.correlate2d(
                    self.input[:, :, d], output_error[:, :, k], 'valid')
            dBias[k] = self.layer_depth * np.sum(output_error[:, :, k])

        self.weights -= learning_rate*dWeights
        self.bias -= learning_rate*dBias
        return in_error


class SoftmaxLayer:
    def __init__(self, input_size):
        self.input_size = input_size

    def forward(self, input):
        self.input = input
        tmp = np.exp(input)
        self.output = tmp / np.sum(tmp)
        return self.output

    def backward(self, output_error, learning_rate):
        input_error = np.zeros(output_error.shape)
        out = np.tile(self.output.T, self.input_size)
        return self.output * np.dot(output_error, np.identity(self.input_size) - out)


class Network:

    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    def add(self, layer):
        """ obviously, add a layer to network """
        self.layers.append(layer)

    def use(self, loss, loss_prime):
        """ set loss to use """
        self.loss = loss
        self.loss_prime = loss_prime

    def predict(self, input_data):
        """ pridict output for given input  """
        samples = len(input_data)
        result = []

        # run network over all samples
        for i in range(samples):

            # forward propagation
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)

        return result

    def fit(self, x_train, y_train, epochs, learning_rate):
        """ training the network  """
        samples = len(x_train)

        # train loop
        for i in range(epochs):
            err = 0
            for j in range(samples):

                # forward propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # compute loss
                err += self.loss(y_train[j], output)

                # backward propagation
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            # claculate average error on all samples
            err /= samples
            # print("epoch {}/{} - error={}".format(i+1, epochs, err))

            print('epoch %d/%d   error=%f' % (i+1, epochs, err))

# XOR EXAMPLE


# train data
x_train = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

# network
net = Network()
net.add(FCLayer(2, 3))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(3, 1))
net.add(ActivationLayer(tanh, tanh_prime))

# train model
net.use(mse, mse_prime)
net.fit(x_train, y_train, epochs=1000, learning_rate=0.1)

# test predict
out = net.predict(x_train)
print(out)


# print('epoch %d/%d   error=%f' % (i+1, epochs, err))


# MNIST EXAMPLE

# load MNIST from server
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# training data : 60000 samples
# reshape and normalize input data
x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
x_train = x_train.astype('float32')
x_train /= 255
# encode output which is a number in range [0,9] into a vector of size 10
# e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
y_train = np_utils.to_categorical(y_train)

# same for test data : 10000 samples
x_test = x_test.reshape(x_test.shape[0], 1, 28*28)
x_test = x_test.astype('float32')
x_test /= 255
y_test = np_utils.to_categorical(y_test)

# Network
net = Network()
# input_shape=(1, 28*28)    ;   output_shape=(1, 100)
net.add(FCLayer(28*28, 100))
net.add(ActivationLayer(tanh, tanh_prime))
# input_shape=(1, 100)      ;   output_shape=(1, 50)
net.add(FCLayer(100, 50))
net.add(ActivationLayer(tanh, tanh_prime))
# input_shape=(1, 50)       ;   output_shape=(1, 10)
net.add(FCLayer(50, 10))
net.add(ActivationLayer(tanh, tanh_prime))

# train on 1000 samples
# as we didn't implemented mini-batch GD, training will be pretty slow if we update at each iteration on 60000 samples...
net.use(mse, mse_prime)
net.fit(x_train[0:1000], y_train[0:1000], epochs=35, learning_rate=0.1)

# test on 3 samples
out = net.predict(x_test[0:3])
print("\n")
print("predicted values : ")
print(out, end="\n")
print("true values : ")
print(y_test[0:3])
