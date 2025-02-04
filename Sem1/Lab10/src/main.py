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
    dt = 0.001
    data = Model.harm()
    shifted_data = Model.addModel(data, 10)
    unshifted_data = Processing.antiShift(data)

    plt.subplot(1, 3, 1)
    plt.ylim(-120, 120)
    make_line_plot("Исходный процесс", np.arange(len(data)) * dt, data,
                   'Time [sec]', 'Amplitude')
    plt.subplot(1, 3, 2)
    plt.ylim(-120, 120)
    make_line_plot("Смещённый процесс", np.arange(len(data)) * dt, shifted_data,
                   'Time [sec]', 'Amplitude')
    plt.subplot(1, 3, 3)
    plt.ylim(-120, 120)
    make_line_plot("Исправленный процесс", np.arange(len(data)) * dt, unshifted_data,
                   'Time [sec]', 'Amplitude')

def __gen_procs(N=1000, R=30, M=10, type='noise'):
    data = list()
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
            result = Processing.antiNoise(noise, M[i], N)
            plt.subplot(2, 3, i+1)
            plt.ylim(-32, 32)
            make_line_plot(f"Noise vector; M = {M[i]}, std = {np.std(result):.4f}", np.arange(N) * dt,
                           result, "Time [sec]", "Amplitude")

    def __noise_harm_add():
        for i in range(len(M)):
            noise = __gen_procs(N, R, M[i], type='noise')
            harm = __gen_procs(N, R, M[i], type='harm')
            add_model = []
            for j in range(M[i]):
                add_model.append(Model.addModel(noise[j], harm[j]))
            result = Processing.antiNoise(np.array(add_model), M[i], N)
            plt.subplot(2, 3, i+1)
            plt.ylim(-42, 42)
            make_line_plot(f"Noise + harm; M = {M[i]}, std = {np.std(result):.4f}", np.arange(N) * dt,
                           result, "Time [sec]", "Amplitude")

    N = 1000
    M = [1, 10, 100, 1000, 10000]
    R = 30
    dt = 0.001
    inner_funcs = [__noise_vector, __noise_harm_add]
    for i in range(len(inner_funcs)):
        inner_funcs[i]()
        if i != len(inner_funcs) - 1:
            plt.tight_layout()
            fig.savefig(f"./plots/{title}_{i+1}.png")
            current_title = fig._suptitle.get_text()
            fig = plt.figure(figsize=(14, 8))
            fig.suptitle(current_title)

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
    make_line_plot("Dependence of \u03C3 from M", [i for i in range(1,M+1,10)],
                   std_devs, "M", "\u03C3")

def test_snr_method():
    N = 1000
    M = [1, 10, 100, 1000, 10000]
    R = 30
    for i in range(len(M)):
        noise = __gen_procs(N, R, M[i], type='noise')
        harm = __gen_procs(N, R, M[i], type='harm')
        add_model = []
        for j in range(M[i]):
            add_model.append(Model.addModel(noise[j], harm[j]))
        denoised_sig = Processing.antiNoise(np.array(add_model), M[i], N)
        noise = Processing.antiNoise(noise, M[i], N)
        result = Analysis.SNR(denoised_sig, noise)

        print(f'Отношение сигнал/шум (SNR) для M={M[i]}:    {result}')


if __name__ == '__main__':
    sns.set_theme(style="whitegrid")
    if not os.path.exists("./plots"):
        os.makedirs("./plots")

    test_methods = {
        test_antiShift_method: 'Anti-shift method',
        test_antiNoise_method: 'Anti-noise method',
        test_std_div_from_amount_dep: 'Std deviation from M dependency',
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
