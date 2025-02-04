import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from processing import FMModulator
from src.analysis import Analysis
from src.model import Model


def make_line_plot(title: str, x: np.ndarray, y: np.ndarray, x_label="Time [sec]", y_label="Amplitude"):
    plot = sns.lineplot(x=x, y=y)
    plot.set_xlabel(x_label, fontsize=12)
    plot.set_ylabel(y_label, fontsize=12)
    plt.title(title)
    return plt

def read_data(filename):
    with open(filename, 'rb') as f:
        data = np.fromfile(f, dtype=np.float32)
    return data


def plot_harm_process():
    N = 1_000
    rate = 1_000
    duration = N / rate
    f_m = 5
    A = 100

    dt = 1 / rate
    # deltaF = (1/dt) / N
    # f_bound = 1 / (2*dt)

    harm = Model.harm(N, A, f_m, dt)
    fft_result = np.fft.fft(harm)
    fft_freq = np.fft.fftfreq(N, d=dt)
    amplitude_spectrum = np.abs(fft_result) / N
    # Re, Im = Analysis.fourier(harm, N)
    # specter = Analysis.spectrFourier(Re, Im, N//2)

    print(f'Длительность гармонического процесса: {duration} сек')

    plt.subplot(2, 1, 1)
    make_line_plot(f'Гармонический процесс; f={f_m} Гц, rate={rate} Гц', np.arange(N) * dt,
                   harm, 'Time [sec]', 'Amplitude')
    plt.subplot(2, 1, 2)
    make_line_plot('Амплитудный спектр Фурье', fft_freq[:N//2],
                   amplitude_spectrum[:N//2], 'Freq [Hz]', '|X_n|')
    # make_line_plot('Амплитудный спектр Фурье', np.arange(0, f_bound, deltaF),
    #                specter, 'Freq [Hz]', '|X_n|')

def plot_carrier_harm_process():
    N = 1_000
    rate = 1_000
    duration = N / rate
    f_c = 100
    A = 100

    dt = 1 / rate
    # deltaF = (1/dt) / N
    # f_bound = 1 / (2*dt)

    harm = Model.harm(N, A, f_c, dt)
    fft_result = np.fft.fft(harm)
    fft_freq = np.fft.fftfreq(N, d=dt)
    amplitude_spectrum = np.abs(fft_result) / N
    # Re, Im = Analysis.fourier(harm, N)
    # specter = Analysis.spectrFourier(Re, Im, N//2)

    print(f'Длительность гармонического процесса: {duration} сек')

    plt.subplot(2, 1, 1)
    make_line_plot(f'Несущий сигнал; f={f_c} Гц, rate={rate} Гц', np.arange(N) * dt,
                   harm, 'Time [sec]', 'Amplitude')
    plt.subplot(2, 1, 2)
    make_line_plot('Амплитудный спектр Фурье', fft_freq[:N//2],
                   amplitude_spectrum[:N//2], 'Freq [Hz]', '|X_n|')
    # make_line_plot('Амплитудный спектр Фурье', np.arange(0, f_bound, deltaF),
    #                specter, 'Freq [Hz]', '|X_n|')

def plot_FM_signal():
    N = 1_000
    rate = 1_000
    f_m = 5
    f_c = 100
    A_c = 100
    h = 15

    dt = 1 / rate
    # deltaF = (1/dt) / N
    # f_bound = 1 / (2*dt)
    modulator = FMModulator()

    harm = Model.harm(N, A_c, f_m, dt)

    modulated_signal = modulator.modulate(N, A_c, f_c, f_m, h, dt)
    fft_result = np.fft.fft(modulated_signal)
    fft_freq = np.fft.fftfreq(N, d=dt)
    amplitude_spectrum = np.abs(fft_result) / N
    # Re_m, Im_m = Analysis.fourier(modulated_signal, N)
    # specter_m = Analysis.spectrFourier(Re_m, Im_m, N//2)

    plt.subplot(3, 1, 1)
    make_line_plot(f'Модулирующий сигнал; f={f_m} Гц, rate={rate} Гц', np.arange(N) * dt,
                   harm, 'Time [sec]', 'Amplitude')
    plt.subplot(3, 1, 2)
    make_line_plot('Частотно-смодулированный сигнал', np.arange(N) * dt,
                   modulated_signal, 'Time [sec]', 'Amplitude')
    plt.subplot(3, 1, 3)
    make_line_plot('Амплитудный спектр Фурье', fft_freq[:N//2],
                   amplitude_spectrum[:N//2], 'Freq [Hz]', '|X_n|')
    # make_line_plot('Спектр частотно-смодулированного сигнала', np.arange(0, f_bound, deltaF),
    #                specter_m, 'Freq [Hz]', '|X_n|')

def plot_FM_with_various_h():
    N = 1_000
    rate = 1_000
    f_m = 5
    f_c = 100
    A_c = 100
    h = [0.1, 1, 15]
    dt = 1 / rate

    modulator = FMModulator()
    for i in range(len(h)):
        modulated_signal = modulator.modulate(N, A_c, f_c, f_m, h[i], dt)
        fft_result = np.fft.fft(modulated_signal)
        fft_freq = np.fft.fftfreq(N, d=dt)
        amplitude_spectrum = np.abs(fft_result) / N

        plt.subplot(len(h), 2, i*2+1)
        make_line_plot(f'FM сигнал; h={h[i]}', np.arange(N) * dt,
                       modulated_signal, 'Time [sec]', 'Amplitude')
        plt.subplot(len(h), 2, i*2+2)
        make_line_plot('Амплитудный спектр Фурье', fft_freq[:N//2],
                       amplitude_spectrum[:N//2], 'Freq [Hz]', '|X_n|')

def plot_FM_with_small_noise():
    global fig

    def __random_noise():
        noise = Model.noise(N, R)
        for i in range(len(h)):
            print(f'Random small noise #{i+1}')
            modulated_signal = modulator.modulate(N, A_c, f_c, f_m, h[i], dt)
            print(f'SNR = {Analysis.SNR(modulated_signal, noise):.2f} дБ')
            noised_signal = Model.addModel(noise, modulated_signal)
            fft_result = np.fft.fft(noised_signal)
            fft_freq = np.fft.fftfreq(N, d=dt)
            amplitude_spectrum = np.abs(fft_result) / N

            plt.subplot(len(h), 3, i*3+1)
            make_line_plot(f'Случайный шум; R={R}', np.arange(N) * dt,
                           noise, 'Time [sec]', 'Amplitude')
            plt.subplot(len(h), 3, i*3+2)
            make_line_plot(f'FM сигнал с белым шумом; h={h[i]}', np.arange(N) * dt,
                           noised_signal, 'Time [sec]', 'Amplitude')
            plt.subplot(len(h), 3, i*3+3)
            make_line_plot('Амплитудный спектр Фурье', fft_freq[:N//2],
                           amplitude_spectrum[:N//2], 'Freq [Hz]', '|X_n|')

    def __spike_noise():
        noise = Model.spikes(N, M, R, Rs)
        for i in range(len(h)):
            print(f'Spike small noise #{i+1}')
            modulated_signal = modulator.modulate(N, A_c, f_c, f_m, h[i], dt)
            print(f'SNR = {Analysis.SNR(modulated_signal, noise):.2f} дБ')
            noised_signal = Model.addModel(noise, modulated_signal)
            fft_result = np.fft.fft(noised_signal)
            fft_freq = np.fft.fftfreq(N, d=dt)
            amplitude_spectrum = np.abs(fft_result) / N

            plt.subplot(len(h), 3, i*3+1)
            make_line_plot(f'Импульсный шум; R={R}, M={M}, Rs={Rs}', np.arange(N) * dt,
                           noise, 'Time [sec]', 'Amplitude')
            plt.subplot(len(h), 3, i*3+2)
            make_line_plot(f'FM сигнал с белым шумом; h={h[i]}, R={R}', np.arange(N) * dt,
                           noised_signal, 'Time [sec]', 'Amplitude')
            plt.subplot(len(h), 3, i*3+3)
            make_line_plot('Амплитудный спектр Фурье', fft_freq[:N//2],
                           amplitude_spectrum[:N//2], 'Freq [Hz]', '|X_n|')


    N = 1_000
    R = 1
    M = 1
    Rs = R * 0.1

    rate = 1_000
    f_m = 5
    f_c = 100
    A_c = 100
    h = [0.1, 1, 15]
    dt = 1 / rate

    modulator = FMModulator()
    inner_funcs = [__random_noise, __spike_noise]
    for i in range(len(inner_funcs)):
        inner_funcs[i]()
        if i != len(inner_funcs) - 1:
            plt.tight_layout()
            fig.savefig(f"./plots/{title}_{i+1}.png")
            current_title = fig._suptitle.get_text()
            fig = plt.figure(figsize=(14, 8))
            fig.suptitle(current_title)

def plot_FM_with_big_noise():
    global fig

    def __random_noise():
        noise = Model.noise(N, R)
        for i in range(len(h)):
            print(f'Random small noise #{i+1}')
            modulated_signal = modulator.modulate(N, A_c, f_c, f_m, h[i], dt)
            print(f'SNR = {Analysis.SNR(modulated_signal, noise):.2f} дБ')
            noised_signal = Model.addModel(noise, modulated_signal)
            fft_result = np.fft.fft(noised_signal)
            fft_freq = np.fft.fftfreq(N, d=dt)
            amplitude_spectrum = np.abs(fft_result) / N

            plt.subplot(len(h), 3, i*3+1)
            make_line_plot(f'Случайный шум; R={R}', np.arange(N) * dt,
                           noise, 'Time [sec]', 'Amplitude')
            plt.subplot(len(h), 3, i*3+2)
            make_line_plot(f'FM сигнал с белым шумом; h={h[i]}', np.arange(N) * dt,
                           noised_signal, 'Time [sec]', 'Amplitude')
            plt.subplot(len(h), 3, i*3+3)
            make_line_plot('Амплитудный спектр Фурье', fft_freq[:N//2],
                           amplitude_spectrum[:N//2], 'Freq [Hz]', '|X_n|')

    def __spike_noise():
        noise = Model.spikes(N, M, R_spike, Rs)
        for i in range(len(h)):
            print(f'Spike small noise #{i+1}')
            modulated_signal = modulator.modulate(N, A_c, f_c, f_m, h[i], dt)
            print(f'SNR = {Analysis.SNR(modulated_signal, noise):.2f} дБ')
            noised_signal = Model.addModel(noise, modulated_signal)
            fft_result = np.fft.fft(noised_signal)
            fft_freq = np.fft.fftfreq(N, d=dt)
            amplitude_spectrum = np.abs(fft_result) / N

            plt.subplot(len(h), 3, i*3+1)
            make_line_plot(f'Импульсный шум; R={R}, M={M}, Rs={Rs}', np.arange(N) * dt,
                           noise, 'Time [sec]', 'Amplitude')
            plt.subplot(len(h), 3, i*3+2)
            make_line_plot(f'FM сигнал с белым шумом; h={h[i]}, R={R}', np.arange(N) * dt,
                           noised_signal, 'Time [sec]', 'Amplitude')
            plt.subplot(len(h), 3, i*3+3)
            make_line_plot('Амплитудный спектр Фурье', fft_freq[:N//2],
                           amplitude_spectrum[:N//2], 'Freq [Hz]', '|X_n|')


    N = 1_000
    A_c = 100
    R = int(A_c * 0.75)  # [A_c * 0.25, A_c * 0.5, A_c * 0.75]
    R_spike = 500
    M = 2
    Rs = R_spike * 0.1

    rate = 1_000
    f_m = 5
    f_c = 100
    h = [0.1, 1, 15]
    dt = 1 / rate

    modulator = FMModulator()
    inner_funcs = [__random_noise, __spike_noise]
    for i in range(len(inner_funcs)):
        inner_funcs[i]()
        if i != len(inner_funcs) - 1:
            plt.tight_layout()
            fig.savefig(f"./plots/{title}_{i+1}.png")
            current_title = fig._suptitle.get_text()
            fig = plt.figure(figsize=(14, 8))
            fig.suptitle(current_title)


if __name__ == '__main__':
    sns.set_theme(style='whitegrid')
    if not os.path.exists('./plots'):
        os.makedirs('./plots')

    test_methods = {
        # plot_harm_process: 'Harm process',
        # plot_carrier_harm_process: 'Carrier harm process',
        # plot_FM_signal: 'FM signal',
        # plot_FM_with_various_h: 'FM signal with various h',
        # plot_FM_with_small_noise: 'FM with small noise',
        plot_FM_with_big_noise: 'FM with big noise',
    }

    for test_method, title in test_methods.items():
        fig = plt.figure(figsize=(14, 8))
        fig.suptitle(title)
        test_method()
        plt.tight_layout()
        fig.savefig(f'./plots/{title}.png')
    plt.show()
