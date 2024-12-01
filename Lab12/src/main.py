from typing import Callable

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

from processing import Processing
from src.analysis import Analysis
from src.model import Model, TrendFuncs


def make_line_plot(title: str, x: np.ndarray, y: np.ndarray, x_label="Time [sec]", y_label="Amplitude"):
    plot = sns.lineplot(x=x, y=y)
    plot.set_xlabel(x_label, fontsize=12)
    plot.set_ylabel(y_label, fontsize=12)
    plt.title(title)
    return plt


def test_filters_methods():
    global fig

    def __lpf():
        lpf = Processing.lpf(fc, m, dt)
        tf_lpf = Analysis.transferFunction(lpf, m)
        plt.subplot(1, 2, 1)
        make_line_plot('Импульсная реакция фильтра lpf', np.arange(len(lpf)) * dt,
                       lpf, 'Time [sec]', 'Amplitude')
        plt.subplot(1, 2, 2)
        make_line_plot('Частотная характеристика фильтра lpf', np.arange(len(tf_lpf)) * dt,
                       tf_lpf, 'Time [sec]', 'Amplitude')

    def __hpf():
        hpf = Processing.hpf(fc, m, dt)
        tf_hpf = Analysis.transferFunction(hpf, m)
        plt.subplot(1, 2, 1)
        make_line_plot('Импульсная реакция фильтра hpf', np.arange(len(hpf)) * dt,
                       hpf, 'Time [sec]', 'Amplitude')
        plt.subplot(1, 2, 2)
        make_line_plot('Частотная характеристика фильтра hpf', np.arange(len(tf_hpf)) * dt,
                       tf_hpf, 'Time [sec]', 'Amplitude')

    def __bpf():
        bpf = Processing.lpf(fc, m, dt)
        tf_bpf = Analysis.transferFunction(bpf, m)
        plt.subplot(1, 2, 1)
        make_line_plot('Импульсная реакция фильтра bpf', np.arange(len(bpf)) * dt,
                       bpf, 'Time [sec]', 'Amplitude')
        plt.subplot(1, 2, 2)
        make_line_plot('Частотная характеристика фильтра bpf', np.arange(len(tf_bpf)) * dt,
                       tf_bpf, 'Time [sec]', 'Amplitude')

    def __bsf():
        bsf = Processing.lpf(fc, m, dt)
        tf_bsf = Analysis.transferFunction(bsf, m)
        plt.subplot(1, 2, 1)
        make_line_plot('Импульсная реакция фильтра bsf', np.arange(len(bsf)) * dt,
                       bsf, 'Time [sec]', 'Amplitude')
        plt.subplot(1, 2, 2)
        make_line_plot('Частотная характеристика фильтра bsf', np.arange(len(tf_bsf)) * dt,
                       tf_bsf, 'Time [sec]', 'Amplitude')

    fc, fc1, fc2 = 50, 35, 75
    dt = 0.002
    m = 64
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
        test_filters_methods: 'Filters methods'
    }

    for test_method, title in test_methods.items():
        fig = plt.figure(figsize=(14, 8))
        fig.suptitle(title)
        test_method()
        plt.tight_layout()
        fig.savefig(f"./plots/{title}.png")
    plt.show()
