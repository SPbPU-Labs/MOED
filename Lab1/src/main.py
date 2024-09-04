import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from src.model import Model, TrendFuncs

a_linear = 3
a_exp = 0.008
b = 0
delta = 1
N = 1000

if __name__ == '__main__':
    model = Model()

    linear_up_data = model.trend(TrendFuncs.linear_func, a_linear, b, delta, N)
    linear_down_data = model.trend(TrendFuncs.linear_func, -a_linear, b, delta, N)
    exponential_up_data = model.trend(TrendFuncs.exp_func, -a_exp, b, delta, N)
    exponential_down_data = model.trend(TrendFuncs.exp_func, a_exp, b, delta, N)

    sns.set(style="whitegrid")
    plt.figure(figsize=(14, 8))

    plt.subplot(2, 2, 1)
    sns.lineplot(x=np.arange(N), y=linear_up_data)
    plt.title('Linear Uptrend')

    plt.subplot(2, 2, 2)
    sns.lineplot(x=np.arange(N), y=linear_down_data)
    plt.title('Linear Downtrend')

    plt.subplot(2, 2, 4)
    sns.lineplot(x=np.arange(N), y=exponential_down_data)
    plt.title('Exponential Downtrend')

    plt.subplot(2, 2, 3)
    sns.lineplot(x=np.arange(N), y=exponential_up_data)
    plt.title('Exponential Uptrend')

    plt.tight_layout()
    plt.show()
