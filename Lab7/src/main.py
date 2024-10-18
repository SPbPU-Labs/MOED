import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

from analysis import Analysis
from src.model import Model, TrendFuncs


def make_line_plot(title: str, x: np.ndarray, y: np.ndarray, x_label="N", y_label="x(t)"):
    plot = sns.lineplot(x=x, y=y)
    plot.set_xlabel(x_label, fontsize=12)
    plot.set_ylabel(y_label, fontsize=12)
    plt.title(title)
    N = len(x)
    plt.xticks(np.arange(0, N + N//10, N//10))
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    return plot

def test_spectrFourier_method():
    def __test_harm_method():
        A0 = 100
        f = 115
        dt = 0.001
        data = Model.harm(N, A0, f, dt)
        Re, Im = Analysis.fourier(data, N)
        result = Analysis.spectrFourier(Re, Im, N_half, Dt)
        plt.subplot(4, 2, 1)
        make_line_plot('Harm', np.arange(N_half), result, 'f', '|X_n|')

    def __test_polyHarm_method():
        m = 3
        A = [100, 15, 20]
        f = [33, 5, 170]
        dt = 0.002
        data = Model.polyHarm(N, A, f, m, dt)
        Re, Im = Analysis.fourier(data, N)
        result = Analysis.spectrFourier(Re, Im, N_half, Dt)
        plt.subplot(4, 2, 2)
        make_line_plot('PolyHarm', np.arange(N_half), result, 'f', '|X_n|')
        print(f'Частота полигармонического процесса: {np.max(f)}')

    def __test_noise_method():
        R = 100
        data = Model.noise(N, R)
        Re, Im = Analysis.fourier(data, N)
        result = Analysis.spectrFourier(Re, Im, N_half, Dt)
        plt.subplot(4, 2, 3)
        make_line_plot('Noise', np.arange(N_half), result, 'f', '|X_n|')

    def __test_spikes_method():
        M = 10
        R = 100
        Rs = 100 * 0.1
        data = Model.spikes(N, M, R, Rs)
        Re, Im = Analysis.fourier(data, N)
        result = Analysis.spectrFourier(Re, Im, N_half, Dt)
        plt.subplot(4, 2, 4)
        make_line_plot('Spikes', np.arange(N_half), result, 'f', '|X_n|')

    def __test_trend_methods():
        a = 10
        b = 0
        delta = 1
        funcs = [TrendFuncs.linear_func, TrendFuncs.exp_func]
        for i in range(len(funcs)):
            data = Model.trend(funcs[i], a, b, delta, N)
            Re, Im = Analysis.fourier(data, N)
            result = Analysis.spectrFourier(Re, Im, N_half, Dt)
            plt.subplot(4, 2, i+5)
            make_line_plot(f'Trend: {funcs[i].__name__}', np.arange(N_half), result, 'f', '|X_n|')

    def __test_constant_func():
        C = [100 for _ in range(N)]
        Re, Im = Analysis.fourier(C, N)
        result = Analysis.spectrFourier(Re, Im, N_half, Dt)
        plt.subplot(4, 2, 7)
        make_line_plot('Const: ', np.arange(N_half), result, 'f', '|X_n|')

    N = 1000
    N_half = N // 2
    Dt = 1
    inner_funcs = [__test_harm_method, __test_polyHarm_method, __test_noise_method,
                   __test_spikes_method, __test_trend_methods, __test_constant_func]
    for func in inner_funcs:
        func()

def test_spectrFourier_method_mult():
    def __test_harm_method():
        A0 = 100
        f = 115
        dt = 0.001
        data = Model.harm(N, A0, f, dt)
        for i in range(len(L)):
            data[N-1-L[i]:] = 0
            Re, Im = Analysis.fourier(data, N)
            result = Analysis.spectrFourier(Re, Im, N_half, Dt)
            plt.subplot(2, 3, i+1)
            make_line_plot(f'Harm; L={L[i]}', np.arange(N_half), result, 'f', '|X_n|')

    def __test_polyHarm_method():
        m = 3
        A = [100, 15, 20]
        f = [33, 5, 170]
        dt = 0.002
        data = Model.polyHarm(N, A, f, m, dt)
        for i in range(len(L)):
            data[N-1-L[i]:] = 0
            Re, Im = Analysis.fourier(data, N)
            result = Analysis.spectrFourier(Re, Im, N_half, Dt)
            plt.subplot(2, 3, i+4)
            make_line_plot(f'PolyHarm; L={L[i]}', np.arange(N_half), result, 'f', '|X_n|')
        print(f'Частота полигармонического процесса: {np.max(f)}')

    N = 1024
    L = [24, 124, 224]
    N_half = N // 2
    Dt = 1
    inner_funcs = [__test_harm_method, __test_polyHarm_method]
    for func in inner_funcs:
        func()

if __name__ == '__main__':
    sns.set_theme(style="whitegrid")
    if not os.path.exists("./plots"):
        os.makedirs("./plots")

    test_methods = {
        test_spectrFourier_method: 'Спектр Фурье',
        test_spectrFourier_method_mult: 'Спектр Фурье (умноженный)'
    }

    for test_method, title in test_methods.items():
        fig = plt.figure(figsize=(14, 8))
        fig.suptitle(title)
        test_method()
        plt.tight_layout()
        fig.savefig(f"./plots/{title}.png")
    plt.show()
