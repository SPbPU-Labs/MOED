import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.model import Model


def make_line_plot(title: str, x: np.ndarray, y: np.ndarray, x_label="Time [sec]", y_label="Amplitude"):
    plot = sns.lineplot(x=x, y=y)
    plot.set_xlabel(x_label, fontsize=12)
    plot.set_ylabel(y_label, fontsize=12)
    plt.title(title)
    return plt


def test_heartRate_method():
    global fig

    def __plot_h():
        make_line_plot('Импульсная реакция h(t)', np.arange(len(h)) * dt, h, 'Time [sec]', 'Amplitude')

    def __plot_x():
        make_line_plot('Управляющая функция x(t)', np.arange(len(x)) * dt, x, 'Time [sec]', 'Amplitude')

    def __plot_y():
        make_line_plot('Кардиограмма y(t)', np.arange(len(y)) * dt, y, 'Time [sec]', 'Amplitude')


    dt = 0.005
    h, x, y = Model.heartRate(dt=dt)
    inner_funcs = [__plot_h, __plot_x, __plot_y]
    for i in range(len(inner_funcs)):
        inner_funcs[i]()
        if i != len(inner_funcs) - 1:
            plt.tight_layout()
            fig.savefig(f"./plots/{title}_{i+1}.png")
            current_title = fig._suptitle.get_text()
            fig = plt.figure(figsize=(14, 8))
            fig.suptitle(current_title)

def test_badHeartRate_method():
    dt = 0.005
    h, x, y = Model.badHeartRate(dt=dt)
    plt.subplot(1, 2, 1)
    make_line_plot('Управляющая функция x(t)', np.arange(len(x)) * dt, x, 'Time [sec]', 'Amplitude')
    plt.subplot(1, 2, 2)
    make_line_plot('Кардиограмма y(t) с паталогиями', np.arange(len(y)) * dt, y, 'Time [sec]', 'Amplitude')


if __name__ == '__main__':
    sns.set_theme(style="whitegrid")
    if not os.path.exists("./plots"):
        os.makedirs("./plots")

    test_methods = {
        # test_heartRate_method: 'HeartRate method',
        test_badHeartRate_method: 'BadHeartRate method',
    }

    for test_method, title in test_methods.items():
        fig = plt.figure(figsize=(14, 8))
        fig.suptitle(title)
        test_method()
        plt.tight_layout()
        fig.savefig(f"./plots/{title}.png")
    plt.show()
