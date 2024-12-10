import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

from processing import Processing
from src.analysis import Analysis
from src.in_out import InOut
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
    make_line_plot('Записанный голос', np.arange(len(audio_data)) / rate,
                   audio_data, 'Counts', 'Amplitude')

def test_change_accent():
    filename = './recorded_voice.wav'
    audio_data, rate, N = InOut.readWAV(filename)
    duration = N / rate
    print(f"Частота дискретизации: {rate} Гц, Длина сигнала: {N} отсчётов, Длительность: {duration:.2f} сек")

    n = [int(rate*0.37), int(rate*0.55), int(rate*0.6), int(rate*0.8)]
    c = [0.5, 4]
    window = rw(c, n, N)
    modified_audio = Model.multModel(audio_data, window)
    InOut.writeWAV('./modified_voice.wav', modified_audio, rate)

    plt.subplot(1, 2, 1)
    plt.ylim(-8200, 6000)
    make_line_plot('Исходный фрагмент', np.arange(len(audio_data)) / rate,
                   audio_data, 'Time [sec]', 'Amplitude')
    plt.subplot(1, 2, 2)
    plt.ylim(-8200, 6000)
    make_line_plot('Фрагмент с изменённым ударением', np.arange(len(modified_audio)) / rate,
                   modified_audio, 'Time [sec]', 'Amplitude')


if __name__ == '__main__':
    sns.set_theme(style="whitegrid")
    if not os.path.exists("./plots"):
        os.makedirs("./plots")

    test_methods = {
        # test_record_audio: 'Audio recording',
        test_change_accent: 'Accent change',
    }

    for test_method, title in test_methods.items():
        fig = plt.figure(figsize=(14, 8))
        fig.suptitle(title)
        test_method()
        plt.tight_layout()
        fig.savefig(f"./plots/{title}.png")
    plt.show()
