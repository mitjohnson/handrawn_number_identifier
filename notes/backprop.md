# Backpropigation

Now that we have an understanding of gradient descent and how to calculate cost functions, how can we apply these changes to each of our layers?

The process of calculating the cost and then adjusting the weights and biases based on our calculated cost is called backpropagation. Backpropagation is an algorithm that, for a single training example, determines how much we need to adjust our weights and biases to minimize the cost for that example effectively.

```python
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
```

While this is very helpful, backpropagation on its own is not sufficient, as it only provides guidance on what to change for a single example.

To improve the overall performance of the network, we would need to perform backpropagation for thousands of training examples and calculate the average of the desired changes. This way, we can make adjustments that affect the overall network performance rather than just a single example. However, performing backpropagation on the entire dataset can be slow.

A more common approach is to split the training data into smaller chunks, commonly referred to as mini-batches. We perform backpropagation on these smaller sets and make adjustments based on the feedback obtained from each mini-batch. This method is a more efficient way to advance towards a local minimum.

This is called Stochastic Gradient Descent (SGD), and works by shuffling test data into smaller chunks and calculating the gradient descent for each training example in this small batch and averaging the results. This may not be the true grasient descent but gives a close enough answer for us to estimate the overall descent and make an adjustment much quicker than the slow backpropigation of possibly millions of test examples.

Example SGD from [network.py](../network.py):

```python
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
```

This is explained in great detail (including the related math) [here](http://neuralnetworksanddeeplearning.com/chap2.html) and [here](https://www.youtube.com/watch?v=tIeHLnjs5U8)
