"""Compare our gradients with PyTorch's on the same expression."""
import torch

from engine import Value


def f(a, b):  # works for both Value and torch tensors
    c = a * b + b ** 2
    return (c * a).tanh() + a / b - c.relu() * 0.5


xa, xb = Value(2.0), Value(-3.0)
f(xa, xb).backward()

ta = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
tb = torch.tensor(-3.0, dtype=torch.float64, requires_grad=True)
f(ta, tb).backward()

for name, mine, ref in (("a", xa.grad, ta.grad.item()), ("b", xb.grad, tb.grad.item())):
    assert abs(mine - ref) < 1e-9, f"d/d{name}: ours {mine} vs torch {ref}"
    print(f"d/d{name}: ours {mine:.6f}  torch {ref:.6f}")
print("gradients match PyTorch")
