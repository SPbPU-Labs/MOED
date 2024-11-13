import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

from analysis import Analysis, ACFType, CCFType
from src.model import Model, TrendFuncs


def make_line_plot(title: str, x: np.ndarray, y: np.ndarray, x_label="N", y_label="x(t)"):
    plot = sns.lineplot(x=x, y=y)
    plot.set_xlabel(x_label, fontsize=12)
    plot.set_ylabel(y_label, fontsize=12)
    plt.title(title)
    return plot

def test_hist_method():
    global fig

    def __test_trend_methods():
        a = 0.001
        b = 0
        delta = 1
        funcs = [TrendFuncs.linear_func, TrendFuncs.exp_func]
        for i in range(len(funcs)):
            data = Model.trend(funcs[i], a, b, delta, N)
            result = Analysis.hist(data, M)
            plt.subplot(2, 2, i*2+1)
            make_line_plot(f'Trend: {funcs[i].__name__}', np.arange(N), data, 'Time [sec]', 'x(t)')
            plt.subplot(2, 2, i*2+2)
            make_line_plot(f'Histogram: {funcs[i].__name__}', np.arange(M), result, 'M', 'h')

    def __test_noise_methods():
        R = 100
        funcs = [Model.noise, Model.myNoise]
        for i in range(len(funcs)):
            data = funcs[i](N, R)
            result = Analysis.hist(data, M)
            plt.subplot(2, 2, i*2+1)
            make_line_plot(f'Noise: {funcs[i].__name__}', np.arange(N), data, 'N', 'x_k')
            plt.subplot(2, 2, i*2+2)
            make_line_plot(f'Histogram: {funcs[i].__name__}', np.arange(M), result, 'M', 'h')

    def __test_harm_methods():
        A0 = 100
        f = 5
        dt = 0.001
        m = 3
        A = [100, 15, 20]
        F = [33, 5, 170]

        time = np.arange(0, N*dt, dt)
        data = Model.harm(N, A0, f, dt)
        result = Analysis.hist(data, M)
        plt.subplot(2, 2, 1)
        make_line_plot('Harm', time, data, 'Time [sec]', 'Amplitude')
        plt.subplot(2, 2, 2)
        make_line_plot('Histogram', np.arange(M), result, 'M', 'h')

        data = Model.polyHarm(N, A, F, m, dt)
        result = Analysis.hist(data, M)
        plt.subplot(2, 2, 3)
        make_line_plot('PolyHarm', time, data, 'Time [sec]', 'Amplitude')
        plt.subplot(2, 2, 4)
        make_line_plot('Histogram', np.arange(M), result, 'M', 'h')
        print(f'Граничная частота полигармонического процесса: {np.max(F)}')

    def __test_spikes_method():
        R = 100
        Rs = 100 * 0.1
        data = Model.spikes(N, 1, R, Rs)
        result = Analysis.hist(data, M)
        plt.subplot(1, 2, 1)
        make_line_plot('Spikes', np.arange(N), data, 'N', 'R')
        plt.subplot(1, 2, 2)
        make_line_plot('Histogram', np.arange(M), result, 'M', 'h')

    N = 10000
    M = 100
    inner_funcs = [__test_trend_methods, __test_noise_methods, __test_harm_methods, __test_spikes_method]
    for i in range(len(inner_funcs)):
        inner_funcs[i]()
        if i != len(inner_funcs) - 1:
            plt.tight_layout()
            fig.savefig(f"./plots/{title}_{i+1}.png")
            current_title = fig._suptitle.get_text()
            fig = plt.figure(figsize=(14, 8))
            fig.suptitle(current_title)

def test_acf_method():
    global fig
    realizations = {Model.noise: 'noise', Model.myNoise: 'myNoise', Model.harm: 'harm'}
    acf_types = [ACFType.ACF, ACFType.COV]
    N = 1000
    R = 100
    A0 = 100
    F0 = 15
    dt = 0.001
    time = np.arange(0, N*dt, dt)

    j = 1
    for t in acf_types:
        i = 0
        for r, name in realizations.items():
            if r is Model.harm:
                data = r(N, A0, F0, dt)
            else:
                data = r(N, R)
            result = Analysis.acf(data, N, t)

            plt.subplot(3, 2, i*2+1)
            plot_name = f'Data type: {name}'
            make_line_plot(plot_name, time, data, 'Time [sec]', 'Amplitude')
            plt.subplot(3, 2, i*2+2)
            plot_name = f'Func type: {t.name}'
            make_line_plot(plot_name, np.arange(N), result, 'L', 'R(L)')
            i += 1

        if j == 1:
            plt.tight_layout()
            fig.savefig(f"./plots/{title}_{j}.png")
            current_title = fig._suptitle.get_text()
            fig = plt.figure(figsize=(14, 8))
            fig.suptitle(current_title)
        j += 1

def test_ccf_method():
    global fig
    realizations = [Model.noise, Model.myNoise, Model.harm, Model.harm]
    ccf_types = [CCFType.CCF, CCFType.CCOV]
    N = 1000
    R = 100
    A0 = 100
    f0 = 15
    f1 = 30
    dt = 0.001
    time = np.arange(0, N*dt, dt)

    j = 1
    for t in ccf_types:
        i = 0
        for r in realizations:
            if r is Model.harm:
                if i % 2 != 1:
                    dataX = r(N, A0, f0, dt)
                    dataY = r(N, A0, f0, dt)
                    plot_name = f'Func type: {t.name}; Realiz: {r.__name__}; f0=f1={f0}'
                else:
                    dataX = r(N, A0, f0, dt)
                    dataY = r(N, A0, f1, dt)
                    plot_name = f'Func type: {t.name}; Realiz: {r.__name__}; f0={f0}, f1={f1}'
            else:
                dataX = r(N, R)
                dataY = r(N, R)
                plot_name = f'Func type: {t.name}; Realiz: {r.__name__}'
            result = Analysis.ccf(dataX, dataY, N, t)

            plt.subplot(4, 3, i*3+1)
            plotX_name = f'Realiz_X: {r.__name__}'
            make_line_plot(plotX_name, time, dataX, 'Time [sec]', 'Amplitude')

            plt.subplot(4, 3, i*3+2)
            plotY_name = f'Realiz_Y: {r.__name__}'
            make_line_plot(plotY_name, time, dataY, 'Time [sec]', 'Amplitude')

            plt.subplot(4, 3, i*3+3)
            make_line_plot(plot_name, np.arange(N), result, 'L', 'R(L)')
            i += 1

        if j == 1:
            plt.tight_layout()
            fig.savefig(f"./plots/{title}_{j}.png")
            current_title = fig._suptitle.get_text()
            fig = plt.figure(figsize=(14, 8))
            fig.suptitle(current_title)
        j += 1


if __name__ == '__main__':
    sns.set_theme(style="whitegrid")
    if not os.path.exists("./plots"):
        os.makedirs("./plots")

    test_methods = {
        test_hist_method: 'Плотность вероятности (гистограмма)',
        test_acf_method: 'Автокор. (ACF) и ковар. (COV) функции',
        test_ccf_method: 'Кросс-кор. (CCF) и кросс-ковар. (CCOV) функции'
    }

    for test_method, title in test_methods.items():
        fig = plt.figure(figsize=(14, 8))
        fig.suptitle(title)
        test_method()
        plt.tight_layout()
        fig.savefig(f"./plots/{title}.png")
    plt.show()
