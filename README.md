# mini-autograd

A tiny autograd engine and neural net library, built from scratch and trained on handwritten digits.

```python
from engine import Value

a = Value(2.0)
b = Value(-3.0)
c = a * b + b ** 2
c.backward()
print(a.grad, b.grad)   # -3.0, -4.0   (dc/da = b, dc/db = a + 2b)
```

Train an MLP:

```bash
pip install -r requirements.txt
python train.py --epochs 5
python check_grads.py    # compares gradients against PyTorch
pytest
```

## Layout

```
engine.py       Value class with backward()
nn.py           Neuron, Layer, MLP
train.py        training loop (softmax cross-entropy, SGD)
check_grads.py  gradient check vs PyTorch
tests/          finite-difference gradient tests
```

## Results

Test accuracy on 8x8 digits (scikit-learn's `load_digits`): fill in after training.

Training takes a few minutes on CPU because the engine is scalar, so a full 28x28 MNIST run would be far too slow. A tensor version with NumPy is the natural next step.

PyTorch is only used in `check_grads.py` to verify the math. Inspired by Karpathy's micrograd.

MIT
