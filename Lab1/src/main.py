import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from src.model import Model, TrendFuncs

a_linear = 3
a_exp = 0.008
b = 0
delta = 1
N = 1000


def test_trend_method():
    linear_up_data = model.trend(TrendFuncs.linear_func, a_linear, b, delta, N)
    linear_down_data = model.trend(TrendFuncs.linear_func, -a_linear, b, delta, N)
    exponential_up_data = model.trend(TrendFuncs.exp_func, -a_exp, b, delta, N)
    exponential_down_data = model.trend(TrendFuncs.exp_func, a_exp, b, delta, N)

    for data, i in zip([linear_up_data, linear_down_data, exponential_up_data, exponential_down_data], range(1, 5)):
        plt.subplot(2, 2, i)
        line_plot = sns.lineplot(x=np.arange(N), y=data)
        if i % 2 != 0:
            plt.title('Linear Uptrend')
        else:
            plt.title('Linear Downtrend')
        line_plot.set_xlabel('t', fontsize=12)
        line_plot.set_ylabel('x(t)', fontsize=12)


if __name__ == '__main__':
    model = Model()

    sns.set(style="whitegrid")

    for func, i in zip([test_trend_method], range(1,2)):
        plt.figure(figsize=(14, 8)).suptitle(f"Page #{i}")
        func()

    plt.tight_layout()
    plt.show()
