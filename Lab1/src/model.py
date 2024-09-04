import numpy as np

class Model:
    def trend(self, func, a, b, delta, N):
        t = np.arange(0, N * delta, delta)
        return func(t, a, b)

class TrendFuncs:
    @staticmethod
    def linear_func(t, a, b):
        return a * t + b

    @staticmethod
    def exp_func(t, a, b):
        return np.exp(-a * t) + b

