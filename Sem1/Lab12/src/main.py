import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from processing import Processing
from src.analysis import Analysis


def make_line_plot(title: str, x: np.ndarray, y: np.ndarray, x_label="Time [sec]", y_label="Amplitude"):
    plot = sns.lineplot(x=x, y=y)
    plot.set_xlabel(x_label, fontsize=12)
    plot.set_ylabel(y_label, fontsize=12)
    plt.title(title)
    return plt


def test_lpf_filter_method():
    fc = 50
    dt = 0.002
    m = 64
    lpf, lpf2 = Processing.lpf(fc, m, dt)

    plt.subplot(1, 2, 1)
    make_line_plot('lpf для m+1 весов', range(m+1),
                   lpf, 'Time [index]', 'Amplitude')
    plt.subplot(1, 2, 2)
    make_line_plot('lpf для 2m+1 весов', range(2*m+1),
                   lpf2, 'Time [index]', 'Amplitude')

def test_filters_methods():
    global fig

    def __lpf():
        lpf, lpf2 = Processing.lpf(fc, m, dt)
        tf_lpf = Analysis.transferFunction(lpf2)

        plt.subplot(1, 2, 1)
        make_line_plot('Импульсная реакция фильтра lpf', range(2*m+1),
                       lpf2, 'Time [index]', 'Amplitude')
        plt.subplot(1, 2, 2)
        make_line_plot('Частотная характеристика фильтра lpf', np.arange(0, f_bound, deltaF),
                       tf_lpf, 'Freq [Hz]', 'Amplitude')

    def __hpf():
        hpf = Processing.hpf(fc, m, dt)
        tf_hpf = Analysis.transferFunction(hpf)
        plt.subplot(1, 2, 1)
        make_line_plot('Импульсная реакция фильтра hpf', range(2*m+1),
                       hpf, 'Time [sec]', 'Amplitude')
        plt.subplot(1, 2, 2)
        make_line_plot('Частотная характеристика фильтра hpf', np.arange(0, f_bound, deltaF),
                       tf_hpf, 'Freq [Hz]', 'Amplitude')

    def __bpf():
        bpf = Processing.bpf(fc1, fc2, m, dt)
        tf_bpf = Analysis.transferFunction(bpf)
        plt.subplot(1, 2, 1)
        make_line_plot('Импульсная реакция фильтра bpf', range(2*m+1),
                       bpf, 'Time [sec]', 'Amplitude')
        plt.subplot(1, 2, 2)
        make_line_plot('Частотная характеристика фильтра bpf', np.arange(0, f_bound, deltaF),
                       tf_bpf, 'Freq [Hz]', 'Amplitude')

    def __bsf():
        bsf = Processing.bsf(fc1, fc2, m, dt)
        tf_bsf = Analysis.transferFunction(bsf)
        plt.subplot(1, 2, 1)
        make_line_plot('Импульсная реакция фильтра bsf', range(2*m+1),
                       bsf, 'Time [sec]', 'Amplitude')
        plt.subplot(1, 2, 2)
        make_line_plot('Частотная характеристика фильтра bsf', np.arange(0, f_bound, deltaF),
                       tf_bsf, 'Freq [Hz]', 'Amplitude')

    fc, fc1, fc2 = 50, 35, 75
    dt = 0.002
    m = 64
    N = 2 * m
    f_bound = 1 / (2 * dt)
    deltaF = 2 * f_bound / N
    inner_funcs = [__lpf, __hpf, __bpf, __bsf]
    for i in range(len(inner_funcs)):
        inner_funcs[i]()
        if i != len(inner_funcs) - 1:
            plt.tight_layout()
            fig.savefig(f"./plots/{title}_{i+1}.png")
            current_title = fig._suptitle.get_text()
            fig = plt.figure(figsize=(14, 8))
            fig.suptitle(current_title)


if __name__ == '__main__':
    sns.set_theme(style="whitegrid")
    if not os.path.exists("./plots"):
        os.makedirs("./plots")

    test_methods = {
        test_lpf_filter_method: 'Lpf filter method',
        test_filters_methods: 'Filters methods'
    }

    for test_method, title in test_methods.items():
        fig = plt.figure(figsize=(14, 8))
        fig.suptitle(title)
        test_method()
        plt.tight_layout()
        fig.savefig(f"./plots/{title}.png")
    plt.show()
