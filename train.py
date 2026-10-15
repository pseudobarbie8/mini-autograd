"""Train an MLP built on the scalar autograd engine on 8x8 digit images."""
import argparse
import random
import sys

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from engine import Value
from nn import MLP

sys.setrecursionlimit(20000)


def cross_entropy(logits, target):
    m = max(l.data for l in logits)  # subtract max for numerical stability
    exps = [(l - m).exp() for l in logits]
    total = sum(exps, Value(0.0))
    return -(exps[target] / total).log()


def accuracy(model, X, y):
    correct = 0
    for x, t in zip(X, y):
        out = model(x)
        correct += max(range(len(out)), key=lambda i: out[i].data) == t
    return correct / len(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--n-train", type=int, default=500)
    ap.add_argument("--n-test", type=int, default=300)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.3)
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    random.seed(a.seed)
    digits = load_digits()
    X, y = (digits.data / 16.0).tolist(), digits.target.tolist()
    Xtr, Xte, ytr, yte = train_test_split(X, y, train_size=a.n_train, random_state=a.seed)
    Xte, yte = Xte[:a.n_test], yte[:a.n_test]

    model = MLP(64, [a.hidden, 10])
    params = model.parameters()
    print(f"{len(params)} parameters, {len(Xtr)} train / {len(Xte)} test images")

    for epoch in range(a.epochs):
        idx = list(range(len(Xtr)))
        random.shuffle(idx)
        for s in range(0, len(idx), a.batch):
            batch = idx[s:s + a.batch]
            loss = sum((cross_entropy(model(Xtr[i]), ytr[i]) for i in batch), Value(0.0)) * (1 / len(batch))
            for p in params:
                p.grad = 0.0
            loss.backward()
            for p in params:
                p.data -= a.lr * p.grad
        print(f"epoch {epoch + 1}: last batch loss {loss.data:.3f}, test accuracy {accuracy(model, Xte, yte):.3f}")


if __name__ == "__main__":
    main()
