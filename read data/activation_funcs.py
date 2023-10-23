import numpy as np


class sigmoid:
    @classmethod
    def calc(cls, x):
        return 1 / (1 + np.exp(-x))

    @classmethod
    def grad(cls, x):
        """Gradient of Sigmoid function. Assumes output has already passed through sigmoid!"""
        return x * (1 - x)

class relu:
    @classmethod
    def calc(cls, x):
        return max(0, x)

    @classmethod
    def grad(cls, x):
        return 1 if x > 0 else 0


class softmax:  # can be used to convert output to probabilities (in classification)
    @classmethod
    def calc(cls, x):
        e_x = np.exp(x - np.max(x))  # Subtracting np.max(x) for numerical stability
        return e_x / e_x.sum()

    @classmethod
    def grad(cls, x):
        raise NotImplementedError

class sine:
    @classmethod
    def calc(cls, x):
        return np.sin(x)

    @classmethod
    def grad(cls, x):
        return np.cos(x)

class cosine:
    @classmethod
    def calc(cls, x):
        return np.cos(x)

    @classmethod
    def grad(cls, x):
        return -np.sin(x)

class tanh:
    @classmethod
    def calc(cls, x):
        return np.tanh(x)

    @classmethod
    def grad(cls, x):
        return 1 - x**2

class linear:
    @classmethod
    def calc(cls, x):
        return x

    @classmethod
    def grad(cls, x):
        return 1

class gaussian:
    @classmethod
    def calc(cls, x):
        return np.exp(-x**2)

    @classmethod
    def grad(cls, x):
        return -2 * x * np.exp(-x**2)


activation_funcs = {  # pls update with other functions you add
    'sigmoid': sigmoid,
    'relu': relu,
    'softmax': softmax,
    'sine': sine,
    'cosine': cosine,
    'tanh': tanh,
    'linear': linear,
    'gaussian': gaussian,
}
