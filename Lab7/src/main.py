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
    return plt


def test_spectrFourier_method():
    global fig

    def __test_harm_method():
        A0 = 100
        f = 115

        data = Model.harm(N, A0, f, dt)
        Re, Im = Analysis.fourier(data, N)
        result = Analysis.spectrFourier(Re, Im, N//2)

        plt.subplot(1, 2, 1)
        make_line_plot('Harm', time, data, 'Time [sec]', 'Amplitude')
        plt.subplot(1, 2, 2)
        make_line_plot('spectrFourier', np.arange(0, f_bound, deltaF), result, 'Freq [Hz]', '|X_n|')

    def __test_polyHarm_method():
        m = 3
        A = [100, 15, 20]
        f = [33, 5, 170]

        data = Model.polyHarm(N, A, f, m, dt)
        Re, Im = Analysis.fourier(data, N)
        result = Analysis.spectrFourier(Re, Im, N//2)

        plt.subplot(1, 2, 1)
        make_line_plot('polyHarm', time, data, 'Time [sec]', 'Amplitude')
        plt.subplot(1, 2, 2)
        make_line_plot('spectrFourier', np.arange(0, f_bound, deltaF), result, 'Freq [Hz]', '|X_n|')
        print(f'Частота полигармонического процесса: {np.max(f)}')

    def __test_noise_method():
        R = 100

        data = Model.noise(N, R)
        Re, Im = Analysis.fourier(data, N)
        result = Analysis.spectrFourier(Re, Im, N//2)

        plt.subplot(1, 2, 1)
        make_line_plot('Noise', time, data, 'Time [sec]', 'Amplitude')
        plt.subplot(1, 2, 2)
        make_line_plot('spectrFourier', np.arange(0, f_bound, deltaF), result, 'Freq [Hz]', '|X_n|')

    def __test_spikes_method():
        M = 10
        R = 100
        Rs = 100 * 0.1

        data = Model.spikes(N, M, R, Rs)
        Re, Im = Analysis.fourier(data, N)
        result = Analysis.spectrFourier(Re, Im, N//2)

        plt.subplot(1, 2, 1)
        make_line_plot('Spikes', time, data, 'Time [sec]', 'Amplitude')
        plt.subplot(1, 2, 2)
        make_line_plot('spectrFourier', np.arange(0, f_bound, deltaF), result, 'Freq [Hz]', '|X_n|')

    def __test_trend_methods():
        a = 10
        b = 0

        funcs = [TrendFuncs.linear_func, TrendFuncs.exp_func]
        for i in range(len(funcs)):
            data = Model.trend(funcs[i], a, b, dt, N)
            Re, Im = Analysis.fourier(data, N)
            result = Analysis.spectrFourier(Re, Im, N//2)

            plt.subplot(2, 2, i*2+1)
            make_line_plot(f'Trend: {funcs[i].__name__}', time, data, 'Time [sec]', 'Amplitude')
            plt.subplot(2, 2, i*2+2)
            make_line_plot('spectrFourier', np.arange(0, f_bound, deltaF), result, 'Freq [Hz]', '|X_n|')

    def __test_constant_func():
        data = [100 for _ in range(N)]
        Re, Im = Analysis.fourier(data, N)
        result = Analysis.spectrFourier(Re, Im, N//2)

        plt.subplot(1, 2, 1)
        make_line_plot('Const', time, data, 'Time [sec]', 'Amplitude')
        plt.subplot(1, 2, 2)
        make_line_plot('spectrFourier', np.arange(0, f_bound, deltaF), result, 'Freq [Hz]', '|X_n|')

    N = 1000
    dt = 0.001
    deltaF = (1/dt)/N
    f_bound = 1/(2*dt)
    time = np.arange(0, N * dt, dt)

    inner_funcs = [__test_harm_method, __test_polyHarm_method, __test_noise_method,
                   __test_spikes_method, __test_trend_methods, __test_constant_func]
    for i in range(len(inner_funcs)):
        inner_funcs[i]()
        if i != len(inner_funcs) - 1:
            plt.tight_layout()
            fig.savefig(f"./plots/{title}_{i+1}.png")
            current_title = fig._suptitle.get_text()
            fig = plt.figure(figsize=(14, 8))
            fig.suptitle(current_title)

def test_spectrFourier_method_mult():
    global fig

    def __test_harm_method():
        A0 = 100
        f = 115

        data = Model.harm(N, A0, f, dt)
        for i in range(len(L)):
            data[N-1-L[i]:] = 0
            Re, Im = Analysis.fourier(data, N)
            result = Analysis.spectrFourier(Re, Im, N//2)
            plt.subplot(3, 2, i*2+1)
            make_line_plot(f'Harm; L={L[i]}', time, data, 'Time [sec]', 'Amplitude')
            plt.subplot(3, 2, i*2+2)
            make_line_plot('spectrFourier', np.arange(0, f_bound, deltaF), result, 'Freq [Hz]', '|X_n|')

    def __test_polyHarm_method():
        m = 3
        A = [100, 15, 20]
        f = [33, 5, 170]

        data = Model.polyHarm(N, A, f, m, dt)
        for i in range(len(L)):
            data[N-1-L[i]:] = 0
            Re, Im = Analysis.fourier(data, N)
            result = Analysis.spectrFourier(Re, Im, N//2)
            plt.subplot(3, 2, i*2+1)
            make_line_plot(f'polyHarm; L={L[i]}', time, data, 'Time [sec]', 'Amplitude')
            plt.subplot(3, 2, i*2+2)
            make_line_plot('spectrFourier', np.arange(0, f_bound, deltaF), result, 'Freq [Hz]', '|X_n|')
        print(f'Частота полигармонического процесса: {np.max(f)}')

    N = 1024
    L = [24, 124, 224]
    dt = 0.001
    deltaF = (1/dt)/N
    f_bound = 1/(2*dt)
    time = np.arange(0, N * dt, dt)

    inner_funcs = [__test_harm_method, __test_polyHarm_method]
    for i in range(len(inner_funcs)):
        inner_funcs[i]()
        if i != len(inner_funcs) - 1:
            plt.tight_layout()
            fig.savefig(f"./plots/{title}_{i+1}.png")
            current_title = fig._suptitle.get_text()
            fig = plt.figure(figsize=(14, 8))
            fig.suptitle(current_title)

def read_pgp_file():
    file_path = './pgp_dt0005.dat'
    dt = 0.0005
    N = 1000
    deltaF = (1/dt)/N
    f_bound = 1/(2*dt)

    data = np.fromfile(file_path, dtype=np.float32, count=N)
    time = np.arange(0, N * dt, dt)
    Re, Im = Analysis.fourier(data, N)
    result = Analysis.spectrFourier(Re, Im, N//2)

    plt.subplot(1, 2, 1)
    make_line_plot('pgp_dt0005.dat', time, data, 'Time [sec]', 'Amplitude')
    plt.subplot(1, 2, 2)
    make_line_plot('spectrFourier', np.arange(0, f_bound, deltaF), result, 'freq [Hz]', '|X_n|')


if __name__ == '__main__':
    sns.set_theme(style="whitegrid")
    if not os.path.exists("./plots"):
        os.makedirs("./plots")

    test_methods = {
        test_spectrFourier_method: 'Спектр Фурье',
        test_spectrFourier_method_mult: 'Спектр Фурье (умноженный)',
        read_pgp_file: 'Анализ данных из файла'
    }

    for test_method, title in test_methods.items():
        fig = plt.figure(figsize=(14, 8))
        fig.suptitle(title)
        test_method()
        plt.tight_layout()
        fig.savefig(f"./plots/{title}.png")
    plt.show()
