import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from src.model import Model, TrendFuncs


def make_line_plot(title: str, x: np.ndarray, y: np.ndarray):
    plot = sns.lineplot(x=x, y=y)
    plot.set_xlabel('Time [sec]', fontsize=12)
    plot.set_ylabel('Amplitude', fontsize=12)
    plt.title(title)
    return plot

def test_harm_method():
    N = 1000
    A0 = 100
    f = [15 + 100 * i for i in range(6)]
    dt = 0.001
    time = np.arange(0, N*dt, dt)
    for i in range(len(f)):
        plt.subplot(3, 2, i+1)
        data = Model.harm(N, A0, f[i], dt)
        make_line_plot(f'f_0 = {f[i]}', time, data)

def test_polyHarm_method():
    N = 1000
    M = 3
    A = [100, 15, 20]
    f = [33, 5, 170]
    dt = 0.002

    data = Model.polyHarm(N, A, f, M, dt)
    time = np.arange(0, N*dt, dt)
    make_line_plot('', time, data)
    print(f'Частота полигармонического процесса: {np.max(f)}')

def test_addModel_method():
    a = [0.3, 0.05]
    b = [20, 10]
    R = 10
    A = 5
    f = 50
    dt = 0.002
    N = 1000

    time = np.arange(0, N*dt, dt)
    data1 = TrendFuncs.linear_func(time, a[0], b[0])
    data2 = Model.harm(N, A, f, dt)
    result = Model.addModel(data1, data2)  # linear trend + harm
    plt.subplot(2, 3, 1)
    make_line_plot('Linear trend + Harm', time, result)
    plt.subplot(2, 3, 2)
    make_line_plot('Linear trend', time, data1)
    plt.subplot(2, 3, 3)
    make_line_plot('Harm', time, data2)

    data1 = TrendFuncs.exp_func(time, a[1], b[1])
    data2 = Model.noise(N, R)
    result = Model.addModel(data1, data2)  # exp trend + noise
    plt.subplot(2, 3, 4)
    make_line_plot('Exp trend + Noise', time, result)
    plt.subplot(2, 3, 5)
    make_line_plot('Exp trend', time, data1)
    plt.subplot(2, 3, 6)
    make_line_plot('Noise', time, data2)

def test_multModel_method():
    a = [0.3, 0.05]
    b = [20, 10]
    R = 10
    A = 5
    f = 50
    dt = 0.002
    N = 1000

    time = np.arange(0.0, N*dt, dt)
    data1 = TrendFuncs.linear_func(time, a[0], b[0])
    data2 = Model.harm(N, A, f, dt)
    result = Model.multModel(data1, data2)  # linear trend + harm
    plt.subplot(2, 3, 1)
    make_line_plot('Linear trend + Harm', time, result)
    plt.subplot(2, 3, 2)
    make_line_plot('Linear trend', time, data1)
    plt.subplot(2, 3, 3)
    make_line_plot('Harm', time, data2)

    data1 = TrendFuncs.exp_func(time, a[1], b[1])
    data2 = Model.noise(N, R)
    result = Model.multModel(data1, data2)  # exp trend + noise
    plt.subplot(2, 3, 4)
    make_line_plot('Exp trend + Noise', time, result)
    plt.subplot(2, 3, 5)
    make_line_plot('Exp trend', time, data1)
    plt.subplot(2, 3, 6)
    make_line_plot('Noise', time, data2)


if __name__ == '__main__':
    sns.set_theme(style="whitegrid")
    if not os.path.exists("./plots"):
        os.makedirs("./plots")

    test_methods = {
        test_harm_method: 'Гармонический процесс',
        test_polyHarm_method: 'Полигармонический процесс',
        test_addModel_method: 'Аддитивная модель',
        test_multModel_method: 'Мультипликативная модель'
    }
    for test_method, title in test_methods.items():
        fig = plt.figure(figsize=(14, 8))
        fig.suptitle(title)
        test_method()
        plt.tight_layout()
        fig.savefig(f"./plots/{title}.png")
    plt.show()
