import random

from engine import Value


class Neuron:
    def __init__(self, nin, act=True):
        scale = (2 / nin) ** 0.5  # He initialisation
        self.w = [Value(random.gauss(0, scale)) for _ in range(nin)]
        self.b = Value(0.0)
        self.act = act

    def __call__(self, x):
        s = sum((w * xi for w, xi in zip(self.w, x)), self.b)
        return s.relu() if self.act else s

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, nin, nout, **kw):
        self.neurons = [Neuron(nin, **kw) for _ in range(nout)]

    def __call__(self, x):
        return [n(x) for n in self.neurons]

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    def __init__(self, nin, nouts):
        sizes = [nin] + nouts
        self.layers = [
            Layer(sizes[i], sizes[i + 1], act=i < len(nouts) - 1)  # last layer is linear (logits)
            for i in range(len(nouts))
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
