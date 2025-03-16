# How does the MLP train?

Given the explanation of the MultiLayer Perceptron, it may be confusing how this model will "learn."

In the current implementation, given inputs, the MLP will produce seemingly random numbers in the output layer due to the initial random weights assigned to the neurons. We need a way to inform the MLP when its output is incorrect. This is where cost functions come into play.

To calculate the cost, we compute the squared differences between the received output and the expected output. This gives us the cost for a training example. The lower the cost, the better the result; a high cost indicates a worse result. We can take the average cost across all training data to provide a rough idea of how "good" our network is performing.

Now that we have this cost, we need to minimize its value. To do this, we can think of the cost as a function that can be plotted. You might have noticed that this cost function can have thousands of inputs (the weights). To simplify, let's imagine the cost function has one input and one output.

With this function, we can graph the value. Different inputs will correspond to different locations on this graph, and our goal is to find the minimum value. The way to do this is by determining the slope of the line at our current position on the function. Initially, we won't know the slope since we don't have any values to work with, so we will need to provide inputs randomly. Once we have enough information to calculate the slope, we can attempt to find the lowest cost by moving in the direction of the negative slope. This means adjusting our weights based on whether the slope is positive or negative, respectively. This process helps us find the local minimum of our cost function.

Now, unfortunately, finding the minimum cost is a bit more complex than repeatedly finding the slope of a single input function. Instead, our function could have many inputs, creating a kind of surface on top of the x/y axis that resembles a mountain range. Our job is to determine which direction to move on the x/y axis in order to go "down the mountain." (This concept is explained very well [here](https://www.youtube.com/watch?v=IHZwWFHWa-w)).

Now all we need to do is calculate the gradient of our cost function, make an adjustment to the weights using gradient descent, and continue this process until we reach a local minimum cost, with the goal of finding the global minimum. Although this is difficult, and in practice, we may only end up reaching a local minimum, we often have no way of knowing whether we are at a global or local minimum.

This is how we do this in [network.py](../network.py)

```python
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
```

With the output from this cost function, which we hope is at a local or global minimum, we now have meaningful feedback that we can use in our network to adjust the weights. The gradient of the cost function with respect to each weight tells us how much each weight needs to be adjusted to minimize the cost. This information indicates not only whether a weight should increase or decrease but also the magnitude of the adjustment needed (a larger gradient indicates a larger adjustment) and the direction of the change (positive or negative).
