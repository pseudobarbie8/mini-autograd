# mini-autograd

A tiny autograd engine and neural net library, built from scratch and trained on MNIST.

```python
from engine import Value

a = Value(2.0)
b = Value(-3.0)
c = a * b + b ** 2
c.backward()
print(a.grad, b.grad)   # -3.0, -8.0
```

Train an MLP:

```bash
python train.py --epochs 5
python check_grads.py    # compares gradients against PyTorch
```

## Layout

```
engine.py       Value class with backward()
nn.py           Neuron, Layer, MLP
train.py        MNIST training loop
check_grads.py  gradient check vs PyTorch
```

## Results

MNIST test accuracy: fill in after training.

PyTorch is only used in `check_grads.py` to verify the math. Inspired by Karpathy's micrograd.

MIT
