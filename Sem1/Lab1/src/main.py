import os

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from src.model import Model, TrendFuncs

a_linear = 3
a_exp = 0.008
b = 0
delta = 1
N = 1000
C = 200

model = Model()
linear_up_data = model.trend(TrendFuncs.linear_func, a_linear, b, delta, N)
linear_down_data = model.trend(TrendFuncs.linear_func, -a_linear, b, delta, N)
exponential_up_data = model.trend(TrendFuncs.exp_func, -a_exp, b, delta, N)
exponential_down_data = model.trend(TrendFuncs.exp_func, a_exp, b, delta, N)


def test_trend_method():
    for data, i in zip([linear_up_data, linear_down_data, exponential_up_data, exponential_down_data], range(1, 5)):
        plt.subplot(2, 2, i)
        line_plot = sns.lineplot(x=np.arange(N), y=data)
        plt.title(__check_trend(i))
        line_plot.set_xlabel('t', fontsize=12)
        line_plot.set_ylabel('x(t)', fontsize=12)

def test_shift_method():
    for data, i in zip([linear_up_data, linear_down_data, exponential_up_data, exponential_down_data], range(1, 5)):
        data = model.shift(data, N, C)
        plt.subplot(2, 2, i)
        line_plot = sns.lineplot(x=np.arange(N), y=data)
        plt.title(__check_trend(i))
        line_plot.set_xlabel('t', fontsize=12)
        line_plot.set_ylabel('x(t)', fontsize=12)

def test_mult_method():
    for data, i in zip([linear_up_data, linear_down_data, exponential_up_data, exponential_down_data], range(1, 5)):
        data = model.mult(data, N, C)
        plt.subplot(2, 2, i)
        line_plot = sns.lineplot(x=np.arange(N), y=data)
        plt.title(__check_trend(i))
        line_plot.set_xlabel('t', fontsize=12)
        line_plot.set_ylabel('x(t)', fontsize=12)

def piecewise_nonlinear_func():
    """
               { e^(-at) * N,  t < 250, a > 0, N = 1000
               { at,           250 <= t < 500, a < 0
        x(t) = { at,           500 <= t < 750, a > 0
               { e^(-at),      750 <= t < 1000, a < 0
    """
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle("Page #1_1")

    length = int(N / 4)
    y1 = exponential_down_data[:length] * N
    y2 = linear_down_data[length:length*2]
    y3 = linear_up_data[length*2:length*3]
    y4 = exponential_up_data[length*3:length*4]
    y_total = np.concatenate([y1, y2, y3, y4])

    line_plot = sns.lineplot(x=np.arange(N), y=y_total)
    plt.title('Nonlinear func')
    line_plot.set_xlabel('t', fontsize=12)
    line_plot.set_ylabel('x(t)', fontsize=12)
    plt.tight_layout()
    fig.savefig("./plots/fig_1_1.png")

def __check_trend(i: int) -> str:
    if i % 2 != 0:
        if i < 3:
            title = "Linear Uptrend"
        else:
            title = "Exp Uptrend"
    else:
        if i < 3:
            title = "Linear Downtrend"
        else:
            title = "Exp Downtrend"
    return title


if __name__ == '__main__':
    sns.set(style="whitegrid")
    if not os.path.exists("./plots"):
        os.makedirs("./plots")

    for func, i in zip([test_trend_method, test_shift_method, test_mult_method], range(1, 4)):
        fig = plt.figure(figsize=(14, 8))
        fig.suptitle(f"Page #{i}")
        func()
        plt.tight_layout()
        fig.savefig(f"./plots/fig_{i}.png")
    piecewise_nonlinear_func()
    plt.show()
