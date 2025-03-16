"""
network.py

This module defines a basic implementation of a feedforward neural network.
"""

import random
import time
from typing import List, Tuple, Optional, Any

import numpy as np

class Network(object):
    """
    A class to implement a basic neural network.

    Takes one integer value per list indencie, 
    with each indencie representing a layer and the integer
    representing how many neurons the layer has.

    Ex. Three layer MLP with 2 neurons each layer: [2,2,2]

    Allows storage of weights and biases.
    """
    def __init__(self, sizes: List[int]):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    @staticmethod
    def sigmoid(z: np.ndarray):
        """
        Compute the sigmoid activation function.
        """
        return 1.0/(1.0+np.exp(-z))

    @staticmethod
    def sigmoid_prime(z: np.ndarray):
        """
        Compute the derivative of the sigmoid activation function.

        The derivative of the sigmoid function is given by:
            sigmoid_prime(z) = sigmoid(z) * (1 - sigmoid(z))

        This function calculates the derivative for each element in the input array `z`,
        which is useful for backpropagation in neural networks.
        """
        return Network.sigmoid(z)*(1-Network.sigmoid(z))

    def feed_forward(self, a: Any):
        """ Return the output of the network is "a" is input """
        for b, w in zip(self.biases, self.weights):
            a = Network.sigmoid(np.dot(w, a) + b)

        return a

    def sgd(self, training_data: List[Tuple[np.ndarray, np.ndarray]],
        epochs: int, batch_size: int, eta: float,
        test_data: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None):
        """
        Perform Stochastic Gradient Descent (SGD) to train the neural network.

        This method updates the weights and biases of the neural network using the 
        Stochastic Gradient Descent algorithm. The training data is divided into 
        mini-batches, and the model is updated for a specified number of epochs.
        """
        if test_data:
            n_test = len(test_data)

        n = len(training_data)

        for j in range(epochs):
            time1 = time.time()
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+batch_size]
                for k in range(0, n, batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            time2 = time.time()
            if test_data:
                print(
                  f"Epoch {j}: {self.evaluate(test_data)} / {n_test}, in {time2-time1:.2f} seconds"
                )
            else:
                print(f"Epoch {j} complete in {time2-time1:.2f} seconds")

    def update_mini_batch(self, mini_batch: List[Tuple[np.ndarray, np.ndarray]], eta: float):
        """
        Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                        for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x: np.ndarray, y: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Perform backpropagation to compute the gradient of the cost function.

        This method calculates the gradients of the cost function with respect to the 
        weights and biases of the neural network using the backpropagation algorithm.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # Feedforward
        activation = x
        activations = [x]  # List to store all the activations, layer by layer
        zs = []  # List to store all the z vectors, layer by layer

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = Network.sigmoid(z)
            activations.append(activation)

        # Backward pass
        delta = self.cost_derivative(activations[-1], y) * Network.sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = Network.sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())

        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """
        Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation.
        """
        test_results = [(np.argmax(self.feed_forward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """
        Return the vector of partial derivatives partial C_x 
        partial a for the output activations.
        """
        return output_activations - y
