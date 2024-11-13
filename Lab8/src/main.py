import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

from processing import Processing
from src.model import Model, TrendFuncs


def make_line_plot(title: str, x: np.ndarray, y: np.ndarray, x_label="Time [sec]", y_label="Amplitude"):
    plot = sns.lineplot(x=x, y=y)
    plot.set_xlabel(x_label, fontsize=12)
    plot.set_ylabel(y_label, fontsize=12)
    plt.title(title)
    return plt


def test_antiSpike_method():
    global fig

    def __noise_and_spikes_add():
        data1 = Model.noise(N, R)
        data2 = Model.spikes(N, M, 10*R, R_s)
        add_model = Model.addModel(data1, data2)
        result = Processing.antiSpike(add_model, N, R)

        plt.subplot(2, 2, 1)
        make_line_plot('Noise', time, data1)
        plt.subplot(2, 2, 2)
        make_line_plot('Spikes', time, data2)
        plt.subplot(2, 2, 3)
        make_line_plot('addModel', time, add_model)
        plt.subplot(2, 2, 4)
        make_line_plot('antiSpike', time, result)

    def __harm_and_spikes_add():
        data1 = Model.harm(N, A_0)
        data2 = Model.spikes(N, M, 10*R, R_s)
        add_model = Model.addModel(data1, data2)
        result = Processing.antiSpike(add_model, N, R)

        plt.subplot(2, 2, 1)
        make_line_plot('Harm', time, data1)
        plt.subplot(2, 2, 2)
        make_line_plot('Spikes', time, data2)
        plt.subplot(2, 2, 3)
        make_line_plot('addModel', time, add_model)
        plt.subplot(2, 2, 4)
        make_line_plot('antiSpike', time, result)

    N = 1000
    M = 1
    R = 100
    R_s = 100
    A_0 = 100
    dt = 0.001
    time = np.arange(0, N*dt, dt)

    inner_funcs = [__noise_and_spikes_add, __harm_and_spikes_add]
    for i in range(len(inner_funcs)):
        inner_funcs[i]()
        if i != len(inner_funcs) - 1:
            plt.tight_layout()
            fig.savefig(f"./plots/{title}_{i+1}.png")
            current_title = fig._suptitle.get_text()
            fig = plt.figure(figsize=(14, 8))
            fig.suptitle(current_title)

def test_antiTrendLinear_method():
    N = 1000
    A_0 = 5
    f = 50
    a = 0.3
    b = 20
    dt = 0.002
    time = np.arange(0, N*dt, dt)

    data1 = TrendFuncs.linear_func(time, a, b)
    data2 = Model.harm(N, A_0, f, dt)
    add_model = Model.addModel(data1, data2)  # linear trend + harm
    result = Processing.antiTrendLinear(add_model, N)

    plt.subplot(2, 2, 1)
    make_line_plot('Linear trend', time, data1)
    plt.subplot(2, 2, 2)
    make_line_plot('Harm', time, data2)
    plt.subplot(2, 2, 3)
    make_line_plot('addModel', time, add_model)
    plt.subplot(2, 2, 4)
    make_line_plot('antiTrendLinear', time, result)

def test_antiTrendNonLinear_method():
    N = 1000
    W = 10
    R = 100
    a = 0.005
    b = 10
    dt = 0.5

    time = np.arange(0, N*dt, dt)
    data1 = TrendFuncs.exp_func(time, a, b)
    data2 = Model.noise(N, R)
    add_model = Model.addModel(data1, data2)  # linear trend + harm
    # result = Processing.antiTrendNonLinear(add_model, N, W)

    plt.subplot(2, 2, 1)
    make_line_plot('Exp. trend', time, data1)
    plt.subplot(2, 2, 2)
    make_line_plot('Noise', time, data2)
    plt.subplot(2, 2, 3)
    make_line_plot('addModel', time, add_model)
    plt.subplot(2, 2, 4)
    make_line_plot('antiTrendNonLinear', time, data2)


if __name__ == '__main__':
    sns.set_theme(style="whitegrid")
    if not os.path.exists("./plots"):
        os.makedirs("./plots")

    test_methods = {
        # test_antiSpike_method: 'Anti spike',
        # test_antiTrendLinear_method: 'Anti trend linear',
        test_antiTrendNonLinear_method: 'Anti trend non linear'
    }

    for test_method, title in test_methods.items():
        fig = plt.figure(figsize=(14, 8))
        fig.suptitle(title)
        test_method()
        plt.tight_layout()
        fig.savefig(f"./plots/{title}.png")
    plt.show()
