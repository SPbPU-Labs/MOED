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
    def __test_trend_methods():
        a = 10
        b = 0
        delta = 1
        funcs = [TrendFuncs.linear_func, TrendFuncs.exp_func]
        for i in range(len(funcs)):
            data = Model.trend(funcs[i], a, b, delta, N)
            result = Analysis.hist(data, M)
            plt.subplot(3, 3, i+1)
            make_line_plot(f'Trend: {funcs[i].__name__}', np.arange(M), result, 'M', 'h')

    def __test_noise_methods():
        R = 100
        funcs = [Model.noise, Model.myNoise]
        for i in range(len(funcs)):
            data = funcs[i](N, R)
            result = Analysis.hist(data, M)
            plt.subplot(3, 3, i+3)
            make_line_plot(f'Noise: {funcs[i].__name__}', np.arange(M), result, 'M', 'h')

    def __test_spikes_method():
        R = 100
        Rs = 100 * 0.1
        data = Model.spikes(N, M, R, Rs)
        result = Analysis.hist(data, M)
        plt.subplot(3, 3, 5)
        make_line_plot('Spikes', np.arange(M), result, 'M', 'h')

    def __test_harm_method():
        A0 = 100
        f = 15
        dt = 0.001
        data = Model.harm(N, A0, f, dt)
        result = Analysis.hist(data, M)
        plt.subplot(3, 3, 6)
        make_line_plot('Harm', np.arange(M), result, 'M', 'h')

    def __test_polyHarm_method():
        N = 1000
        m = 3
        A = [100, 15, 20]
        f = [33, 5, 170]
        dt = 0.002
        data = Model.polyHarm(N, A, f, m, dt)
        result = Analysis.hist(data, M)
        plt.subplot(3, 3, 7)
        make_line_plot('PolyHarm', np.arange(M), result, 'M', 'h')
        print(f'Частота полигармонического процесса: {np.max(f)}')

    N = 10000
    M = 100
    inner_funcs = [__test_trend_methods, __test_noise_methods, __test_spikes_method,
                   __test_harm_method, __test_polyHarm_method]
    for func in inner_funcs:
        func()

def test_acf_method():
    realizations = [Model.noise, Model.myNoise, Model.harm]
    realizations_name = ['noise', 'myNoise', 'harm']
    acf_types = [ACFType.ACF, ACFType.COV]
    N = 1000
    R = 100
    A0 = 100
    F0 = 15
    dt = 0.001
    i = 0
    for t in acf_types:
        for r in realizations:
            if r is Model.harm:
                data = r(N, A0, F0, dt)
            else:
                data = r(N, R)
            result = Analysis.acf(data, N, t)
            plt.subplot(2, 3, i+1)
            plot_name = f'Func type: {t.name}; Realiz: {realizations_name[i%3]}'
            make_line_plot(plot_name, np.arange(N), result, 'L', 'R(L)')
            i += 1

def test_ccf_method():
    realizations = [Model.noise, Model.myNoise, Model.harm, Model.harm]
    realizations_name = ['noise', 'myNoise', 'harm', 'harm']
    ccf_types = [CCFType.CCF, CCFType.CCOV]
    N = 1000
    R = 100
    A0 = 100
    f0 = 15
    f1 = 30
    dt = 0.001
    i = 0
    for t in ccf_types:
        for r in realizations:
            if r is Model.harm:
                if i % 4 != 3:
                    dataX = r(N, A0, f0, dt)
                    dataY = r(N, A0, f0, dt)
                    plot_name = f'Func type: {t.name}; Realiz: {realizations_name[i%4]}; f0=f1={f0}'
                else:
                    dataX = r(N, A0, f0, dt)
                    dataY = r(N, A0, f1, dt)
                    plot_name = f'Func type: {t.name}; Realiz: {realizations_name[i%4]}; f0={f0}, f1={f1}'
            else:
                dataX = r(N, R)
                dataY = r(N, R)
                plot_name = f'Func type: {t.name}; Realiz: {realizations_name[i%4]}'
            result = Analysis.ccf(dataX, dataY, N, t)
            plt.subplot(4, 2, i+1)
            make_line_plot(plot_name, np.arange(N), result, 'L', 'R(L)')
            i += 1


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
