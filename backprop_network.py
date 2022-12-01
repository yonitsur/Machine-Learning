"""
backprop_network.py
"""

import random
import numpy as np
import math

class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""

        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]


    def SGD(self, training_data, epochs, mini_batch_size, learning_rate, test_data):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  """
        training_accuracy=np.zeros(epochs)
        training_loss=np.zeros(epochs)
        test_accuracy=np.zeros(epochs)
        print("Initial test accuracy: {0}".format(self.one_label_accuracy(test_data)))
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)
            
            training_accuracy[j] = self.one_hot_accuracy(training_data)
            training_loss[j] = self.loss(training_data)
            test_accuracy[j] = self.one_label_accuracy(test_data)
           
            print ("Epoch {0} test accuracy: {1}".format(j, self.one_label_accuracy(test_data)))
        return training_accuracy, training_loss, test_accuracy

   

    def update_mini_batch(self, mini_batch, learning_rate):
        """Update the network's weights and biases by applying
        stochastic gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``."""

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w - (learning_rate / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (learning_rate / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def forward_pass(self, v, z):
        for l in range(self.num_layers-1): 
            v[l]= self.weights[l] @ z[l] + self.biases[l]
            z[l+1] = relu(v[l])

    def backward_pass(self,delta,v):
         for i in reversed(range(self.num_layers-2)):
            delta[i] = np.transpose(self.weights[i+1]) @ delta[i+1] * relu_derivative(v[i])

    def backprop(self, x, y):
        """The function receives as input a 784 dimensional 
        vector x and a one-hot vector y.
        The function should return a tuple of two lists (db, dw) 
        as described in the assignment pdf. """

        delta = [None] * (self.num_layers-1)
        dw = [None] * (self.num_layers-1)
        v = [None] * (self.num_layers-1)
        z = [None] * (self.num_layers)
       
        z[0] = np.copy(x)
        self.forward_pass(v, z)

        delta[-1] = self.loss_derivative_wr_output_activations(v[-1], y)
        self.backward_pass(delta, v)
        
        for i in range(self.num_layers-1):
            dw[i]=np.dot(delta[i], np.transpose(z[i]))

        return delta, dw
    
    def one_label_accuracy(self, data):
        """Return accuracy of network on data with numeric labels"""

        output_results = [(np.argmax(self.network_output_before_softmax(x)), y) for (x, y) in data]
        return sum(int(x == y) for (x, y) in output_results)/float(len(data))


    def one_hot_accuracy(self,data):
        """Return accuracy of network on data with one-hot labels"""

        output_results = [(np.argmax(self.network_output_before_softmax(x)), np.argmax(y)) for (x, y) in data]
        return sum(int(x == y) for (x, y) in output_results)/float(len(data))


    def network_output_before_softmax(self, x):
        """Return the output of the network before softmax if ``x`` is input."""

        layer = 0
        
        for b, w in zip(self.biases, self.weights):
            if layer == len(self.weights) - 1:
                x = np.dot(w, x) + b
            else:
                x = relu(np.dot(w, x)+b)
            layer += 1

        return x


    def loss(self, data):
        """Return the loss of the network on the data"""

        loss_list = []
        for (x, y) in data:
            net_output_before_softmax = self.network_output_before_softmax(x)
            net_output_after_softmax = self.output_softmax(net_output_before_softmax)
            loss_list.append(np.dot(-np.log(net_output_after_softmax).transpose(),y).flatten()[0])

        return sum(loss_list) / float(len(data))


    def output_softmax(self, output_activations):
        """Return output after softmax given output before softmax"""

        output_exp = np.exp(output_activations)
        return output_exp/output_exp.sum()

    def loss_derivative_wr_output_activations(self, output_activations, y):
        """Implement derivative of loss with respect to the output activations before softmax"""

        return self.output_softmax(output_activations- np.max(output_activations)) - y


def relu(z):
    """Return the ReLU of input z"""

    return np.maximum(z,0)



def relu_derivative(z):
    """Implement the derivative of the relu function."""

    return z>0

