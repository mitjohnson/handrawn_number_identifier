# MultiLayer Perceptron

The MLP consists of multiple layers: the input layer, the output layer, and at least one hidden layer. You can have as many hidden layers as you like, and the number of neurons in each layer can also vary.

These layers are composed of neurons, with each neuron connected to multiple neurons in the next layer. The initial layer takes inputs directly and passes them to the first hidden layer.

In the hidden layer, each neuron receives multiple inputs, with each connection having an associated weight. We can calculate a weighted sum based on the activated neurons of the previous layer and the associated weights of the connections.

This weighted sum, in its current state, is not very useful, as it can be any real number. It would be more beneficial to calculate a value that indicates whether a neuron is "activated" or "inactive." This is where an activation function, such as the [sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function), comes into play.

By applying the sigmoid function to our weighted sum, we obtain a value that generally falls between 0 and 1. Values close to 0 result from a negative weighted sum, while values closer to 1 come from a positive weighted sum. This output indicates not only whether the neuron is activated but also "how activated" it is.

However, this alone isn't very useful, as it only tracks whether our value is positive or negative. This is where we need to apply a bias to our weighted sum, allowing us to shift our activation function to the left or right, independent of the input. This adjustment enables us to fine-tune our neurons to identify more complex patterns.
