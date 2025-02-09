import random
from autodiff.engine import Scalar


class Module:
    """
    Module base class
    """

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []


class Neuron(Module):
    """
    Neuron class defining an MLP neuron

    weights     - connection weights to this neuron
    bias        - neuron bias
    activation  - activation function
    """

    def __init__(self, n_inputs, activation='gelu'):
        self.weights = [Scalar(random.uniform(-1, 1)) for _ in range(n_inputs)]
        self.bias = Scalar(0)
        self.activation = activation

    def __call__(self, x):
        y = sum([wi * xi for wi, xi in zip(self.weights, x)], self.bias)

        if self.activation == 'linear':
            return y
        elif self.activation == 'tanh':
            return y.tanh()
        elif self.activation == 'relu':
            return y.relu()
        elif self.activation == 'gelu':
            return y.gelu()

    def parameters(self):
        return self.weights + [self.bias]


class Layer(Module):
    """
    Layer class defining an MLP layer

    neurons     - list of neurons in layer
    """

    def __init__(self, n_inputs, n_outputs, **kwargs):
        self.neurons = [Neuron(n_inputs, **kwargs) for _ in range(n_outputs)]

    def __call__(self, x):
        outputs = [neuron(x) for neuron in self.neurons]
        return outputs[0] if len(outputs) == 1 else outputs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP(Module):
    """
    MLP class defining a multilayer perceptron

    layers      - layers of MLP
    """

    def __init__(self, n_inputs, output_dims, activations):
        layer_sizes = [n_inputs] + output_dims
        self.layers = [Layer(layer_sizes[i], layer_sizes[i + 1], activation=activations[i])
                       for i in range(len(output_dims))]

    def __call__(self, x):
        y = x
        for layer in self.layers:
            y = layer(y)

        return y

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
