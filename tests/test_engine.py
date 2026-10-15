from engine import Value


def f(a, b):
    c = a * b + b ** 2
    return (c * a).tanh() + a / b - c.relu() * 0.5 + (a * 0.1).exp() + (b * b + 1).log()


def test_readme_example():
    a, b = Value(2.0), Value(-3.0)
    (a * b + b ** 2).backward()
    assert (a.grad, b.grad) == (-3.0, -4.0)


def test_gradients_match_finite_differences():
    x, y, h = 2.0, -3.0, 1e-6
    a, b = Value(x), Value(y)
    f(a, b).backward()
    num_a = (f(Value(x + h), Value(y)).data - f(Value(x - h), Value(y)).data) / (2 * h)
    num_b = (f(Value(x), Value(y + h)).data - f(Value(x), Value(y - h)).data) / (2 * h)
    assert abs(a.grad - num_a) < 1e-5
    assert abs(b.grad - num_b) < 1e-5
