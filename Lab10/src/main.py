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


def test_antiShift_method():
    data = Model.harm()
    shifted_data = Model.addModel(data, 10)
    unshifted_data = Processing.antiShift(data)

    plt.subplot(1, 3, 1)
    make_line_plot("Исходный процесс", range(1, 1001), data, 'N', 'Y')
    plt.subplot(1, 3, 2)
    make_line_plot("Смещённый процесс", range(1, 1001), shifted_data, 'N', 'Y')
    plt.subplot(1, 3, 3)
    make_line_plot("Исправленный процесс", range(1, 1001), unshifted_data, 'N', 'Y')

def test_std_div_from_amount_dep():
    N = 1000
    R = 30
    M = 10000
    std_devs = []
    noise = Model.randVector(M, N, R, Model.noise)
    for i in range(1, M+1, 10):
        # Поэлементное осреднение
        avg_noise = np.mean(noise[:i, :], axis=0)
        # Стандартное отклонение осреднённого шума
        std_devs.append(np.std(avg_noise))
    make_line_plot("Dependence of \u03C3 from M", [i for i in range(1,M+1,10)], std_devs, "M", "\u03C3")

def __gen_procs(N=1000, R=30, M=10, type='noise'):
    data = list()
    if M == 1:
        if type == 'noise':
            data = Model.noise(N, R)
        elif type == 'harm':
            data = Model.harm(N, 10, 5, 0.001)
    else:
        if type == 'noise':
            data = Model.randVector(M, N, R, Model.noise)
        elif type == 'harm':
            for i in range(M):
                data.append(Model.harm(N, 10, 5, 0.001))
    return data

def test_antiNoise_method():
    global fig

    def __noise_vector():
        for i in range(len(M)):
            noise = __gen_procs(N, R, M[i], type='noise')
            plt.subplot(2, 3, i+1)
            if M[i] == 1:
                result = Processing.antiNoise(noise, M[i])
                make_line_plot(f"Noise vector; M = {M[i]}", range(1, N+1), result, "N", "x(t)")
            elif M[i] > 1:
                result = Processing.antiNoise(noise, M[i])
                make_line_plot(f"M = {M[i]}", range(1, M[i]+1), result, "M", "x(t)")

    def __noise_harm_add():
        for i in range(len(M)):
            noise = __gen_procs(N, R, M[i], type='noise')
            harm = __gen_procs(N, R, M[i], type='harm')
            add_model = []
            plt.subplot(2, 3, i+1)
            if M[i] == 1:
                add_model = Model.addModel(noise, harm)
                result = Processing.antiNoise(add_model, M[i])
                make_line_plot(f"Harm + Noise; M = {M[i]}", range(1, N+1), result, "N", "x(t)")
            elif M[i] > 1:
                for j in range(M[i]):
                    add_model.append(Model.addModel(noise[j], harm[j]))
                result = Processing.antiNoise(np.array(add_model), M[i])
                make_line_plot(f"M = {M[i]}", range(1, M[i]+1), result, "M", "x(t)")

    N = 1000
    M = [1, 10, 100, 1000, 10000]
    R = 30
    inner_funcs = [__noise_vector, __noise_harm_add]
    for i in range(len(inner_funcs)):
        inner_funcs[i]()
        if i != len(inner_funcs) - 1:
            plt.tight_layout()
            fig.savefig(f"./plots/{title}_{i+1}.png")
            current_title = fig._suptitle.get_text()
            fig = plt.figure(figsize=(14, 8))
            fig.suptitle(current_title)

def test_snr_method():
    N = 1000
    M = [1, 10, 100, 1000, 10000]
    R = 30
    for i in range(len(M)):
        noise = __gen_procs(N, R, M[i], type='noise')
        harm = __gen_procs(N, R, M[i], type='harm')
        add_model = []
        result = None
        if M[i] == 1:
            add_model = Model.addModel(noise, harm)
            std_dev_sig = Processing.antiNoise(add_model, M[i])
            std_dev_noise = Processing.antiNoise(noise, M[i])
            result = Analysis.SNR(std_dev_sig, std_dev_noise)

        elif M[i] > 1:
            for j in range(M[i]):
                add_model.append(Model.addModel(noise[j], harm[j]))
            std_dev_sig = Processing.antiNoise(np.array(add_model), M[i])
            std_dev_noise = Processing.antiNoise(noise, M[i])
            result = Analysis.SNR(std_dev_sig, std_dev_noise)
        print(f'Отношение сигнал/шум (SNR) для M={M[i]}:    {result}')


if __name__ == '__main__':
    sns.set_theme(style="whitegrid")
    if not os.path.exists("./plots"):
        os.makedirs("./plots")

    test_methods = {
        test_antiShift_method: 'Anti-shift method',
        test_std_div_from_amount_dep: 'Std deviation from M dependency',
        test_antiNoise_method: 'Anti-noise method',
        test_snr_method: 'SNR method'
    }

    for test_method, title in test_methods.items():
        if test_method is test_snr_method:
            test_method()
            continue
        fig = plt.figure(figsize=(14, 8))
        fig.suptitle(title)
        test_method()
        plt.tight_layout()
        fig.savefig(f"./plots/{title}.png")
    plt.show()
