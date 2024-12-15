import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.analysis import Analysis
from src.in_out import InOut
from src.model import Model
from src.processing import Processing


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

def rw(c, n, N):
    window = np.ones(N)
    for i in range(1, len(n), 2):
        window[n[i-1]:n[i]] = c[(i-1)//2]
    return window


def test_record_audio():
    filename = './recorded_voice.wav'
    input("Для начала записи голоса нажмите Enter: ")
    print("Началась запись...")
    audio_data, rate, N = InOut.record_audio(filename)
    print("Запись завершена.")
    make_line_plot('Записанное слово "Рама"', np.arange(len(audio_data)) / rate,
                   audio_data, 'Time [sec]', 'Amplitude')

def plot_recorded_word():
    filename = './recorded_voice.wav'
    audio_data, rate, N = InOut.readWAV(filename)
    duration = N / rate
    print(f"Частота дискретизации: {rate} Гц, Длина сигнала: {N} отсчётов, Длительность: {duration:.2f} сек")

    n = [int(rate*0.4), int(rate*1)]
    short_audio_data = audio_data[n[0]:n[1]]
    short_N = len(short_audio_data)
    print(f"Новая длина сигнала: {short_N}")

    dt = 1 / rate
    deltaF = (1/dt)/short_N
    f_bound = 1/(2*dt)
    Re, Im = Analysis.fourier(short_audio_data, short_N)
    result = Analysis.spectrFourier(Re, Im, short_N//2)

    plt.subplot(1, 2, 1)
    make_line_plot('Записанное слово "Рама"', np.arange(len(audio_data)) / rate,
                   audio_data, 'Time [sec]', 'Amplitude')
    plt.subplot(1, 2, 2)
    make_line_plot('Амплитудный спектр', np.arange(0, f_bound, deltaF), result, 'Freq [Hz]', '|X_n|')

def plot_recorded_word_syllables():
    filename = './recorded_voice.wav'
    audio_data, rate, N = InOut.readWAV(filename)
    duration = N / rate
    print(f"Частота дискретизации: {rate} Гц, Длина сигнала: {N} отсчётов, Длительность: {duration:.2f} сек")

    n = [int(rate*0.4), int(rate*0.7), int(rate*0.8), int(rate*1)]
    dt = 1 / rate
    f_bound = 1/(2*dt)

    short_audio_data = audio_data[n[0]:n[1]]
    short_N = len(short_audio_data)
    print(f"Длина сигнала первого слога: {short_N}")

    deltaF = (1/dt)/short_N
    Re, Im = Analysis.fourier(short_audio_data, short_N)
    result = Analysis.spectrFourier(Re, Im, short_N//2)
    # c = [0.5, 4]
    # window = rw(c, n, N)
    # modified_audio = Model.multModel(audio_data, window)
    # InOut.writeWAV('./modified_voice.wav', modified_audio, rate)

    plt.subplot(2, 2, 1)
    plt.ylim(-8200, 6000)
    make_line_plot('Первый слог слова "Рама"', np.arange(len(short_audio_data)) / rate,
                   short_audio_data, 'Time [sec]', 'Amplitude')
    plt.subplot(2, 2, 2)
    make_line_plot('Амплитудный спектр', np.arange(0, f_bound, deltaF), result, 'Freq [Hz]', '|X_n|')

    short_audio_data = audio_data[n[2]:n[3]]
    short_N = len(short_audio_data)
    print(f"Длина сигнала второго слога: {short_N}")

    deltaF = (1/dt)/short_N
    Re, Im = Analysis.fourier(short_audio_data, short_N)
    result = Analysis.spectrFourier(Re, Im, short_N//2)

    plt.subplot(2, 2, 3)
    plt.ylim(-8200, 6000)
    make_line_plot('Второй слог слова "Рама"', np.arange(len(short_audio_data)) / rate,
                   short_audio_data, 'Time [sec]', 'Amplitude')
    plt.subplot(2, 2, 4)
    make_line_plot('Амплитудный спектр', np.arange(0, f_bound, deltaF), result, 'Freq [Hz]', '|X_n|')

def test_filters_methods():
    global fig

    def __lpf():
        lpf, lpf2 = Processing.lpf(fc_lpf, m_lpf, dt)
        tf_lpf = Analysis.transferFunction(lpf2)
        deltaF_lpf = 2 * f_bound / (2 * m_lpf)

        filtered_data = Model.convolModel(lpf2, short_audio_data, N)
        Re, Im = Analysis.fourier(filtered_data, len(filtered_data))
        specter = Analysis.spectrFourier(Re, Im, len(filtered_data)//2)

        plt.subplot(1, 3, 1)
        make_line_plot('Частотная характеристика фильтра lpf', np.arange(0, f_bound, deltaF_lpf),
                       tf_lpf, 'Freq [Hz]', 'Amplitude')
        plt.subplot(1, 3, 2)
        make_line_plot('Данные после фильтра lpf', np.arange(N) * dt,
                       filtered_data, 'Time [index]', 'Amplitude')
        plt.subplot(1, 3, 3)
        make_line_plot('Спектр данных после фильтра lpf', np.arange(0, f_bound, deltaF),
                       specter, 'Freq [Hz]', '|X_n|')

    def __hpf():
        hpf = Processing.hpf(fc_hpf, m_hpf, dt)
        tf_hpf = Analysis.transferFunction(hpf)
        deltaF_hpf = 2 * f_bound / (2 * m_hpf)

        filtered_data = Model.convolModel(hpf, short_audio_data, N)
        Re, Im = Analysis.fourier(filtered_data, N)
        specter = Analysis.spectrFourier(Re, Im, N//2)

        plt.subplot(1, 3, 1)
        make_line_plot('Частотная характеристика фильтра hpf', np.arange(0, f_bound, deltaF_hpf),
                       tf_hpf, 'Freq [Hz]', 'Amplitude')
        plt.subplot(1, 3, 2)
        plt.ylim(-120, 120)
        make_line_plot('Данные после фильтра hpf', np.arange(N) * dt,
                       filtered_data, 'Time [index]', 'Amplitude')
        plt.subplot(1, 3, 3)
        plt.ylim(0, 40)
        make_line_plot('Спектр данных после фильтра hpf', np.arange(0, f_bound, deltaF),
                       specter, 'Freq [Hz]', '|X_n|')

    def __bpf():
        bpf = Processing.bpf(fc1_bpf, fc2_bpf, m_bpf, dt)
        tf_bpf = Analysis.transferFunction(bpf)
        deltaF_bpf = 2 * f_bound / (2 * m_bpf)

        filtered_data = Model.convolModel(bpf, short_audio_data, N)
        Re, Im = Analysis.fourier(filtered_data, N)
        specter = Analysis.spectrFourier(Re, Im, N//2)

        plt.subplot(1, 3, 1)
        make_line_plot('Частотная характеристика фильтра bpf', np.arange(0, f_bound, deltaF_bpf),
                       tf_bpf, 'Freq [Hz]', 'Amplitude')
        plt.subplot(1, 3, 2)
        plt.ylim(-120, 120)
        make_line_plot('Данные после фильтра bpf', np.arange(N) * dt,
                       filtered_data, 'Time [index]', 'Amplitude')
        plt.subplot(1, 3, 3)
        plt.ylim(0, 40)
        make_line_plot('Спектр данных после фильтра bpf', np.arange(0, f_bound, deltaF),
                       specter, 'Freq [Hz]', '|X_n|')


    filename = './recorded_voice.wav'
    audio_data, rate, N = InOut.readWAV(filename)
    duration = N / rate
    print(f"Частота дискретизации: {rate} Гц, Длина сигнала: {N} отсчётов, Длительность: {duration:.2f} сек")

    n = [int(rate*0.4), int(rate*0.7), int(rate*0.8), int(rate*1)]
    dt = 1 / rate
    f_bound = 1/(2*dt)

    short_audio_data = audio_data[n[0]:n[1]]
    short_N = len(short_audio_data)
    print(f"Длина сигнала первого слога: {short_N}")

    deltaF = (1/dt)/short_N

    fc_lpf, fc_hpf, fc1_bpf, fc2_bpf = 15, 200, 15, 200
    m_lpf, m_hpf, m_bpf = 64, 64, 64
    inner_funcs = [__lpf, __hpf, __bpf]
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
        # test_record_audio: 'Audio recording',
        # plot_recorded_word: 'Recorded word _Рама_',
        # plot_recorded_word_syllables: 'The syllables of the word _Рама_',
        test_filters_methods: 'The syllables filtration',
    }

    for test_method, title in test_methods.items():
        fig = plt.figure(figsize=(14, 8))
        fig.suptitle(title)
        test_method()
        plt.tight_layout()
        fig.savefig(f"./plots/{title}.png")
    plt.show()
