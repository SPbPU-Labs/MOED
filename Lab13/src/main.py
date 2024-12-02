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


def test_plot_pgp_dt0005_data():
    filename = './pgp_dt0005.dat'
    data = read_data(filename)
    N = len(data)
    dt = 0.0005
    deltaF = (1/dt)/N
    f_bound = 1/(2*dt)
    time = np.arange(0, N * dt, dt)
    Re, Im = Analysis.fourier(data, N)
    result = Analysis.spectrFourier(Re, Im, N//2)

    plt.subplot(1, 2, 1)
    make_line_plot('pgp_dt0005.dat', time, data, 'Time [sec]', 'Amplitude', )
    plt.subplot(1, 2, 2)
    make_line_plot('Спектр процесса', np.arange(0, f_bound, deltaF), result, 'Freq [Hz]', '|X_n|')

def test_filters_methods():
    global fig

    def __lpf():
        lpf, lpf2 = Processing.lpf(fc_lpf, m_lpf, dt)
        tf_lpf = Analysis.transferFunction(lpf2)
        deltaF_lpf = 2 * f_bound / (2 * m_lpf)

        filtered_data = Model.convolModel(lpf2, data, N)
        Re, Im = Analysis.fourier(filtered_data, len(filtered_data))
        specter = Analysis.spectrFourier(Re, Im, len(filtered_data)//2)

        plt.subplot(1, 3, 1)
        make_line_plot('Частотная характеристика фильтра lpf', np.arange(0, f_bound, deltaF_lpf),
                       tf_lpf, 'Freq [Hz]', 'Amplitude')
        plt.subplot(1, 3, 2)
        plt.ylim(-120, 120)
        make_line_plot('Данные после фильтра lpf', np.arange(N) * dt,
                       filtered_data, 'Time [index]', 'Amplitude')
        plt.subplot(1, 3, 3)
        plt.ylim(0, 40)
        make_line_plot('Спектр данных после фильтра lpf', np.arange(0, f_bound, deltaF),
                       specter, 'Freq [Hz]', '|X_n|')

    def __hpf():
        hpf = Processing.hpf(fc_hpf, m_hpf, dt)
        tf_hpf = Analysis.transferFunction(hpf)
        deltaF_hpf = 2 * f_bound / (2 * m_hpf)

        filtered_data = Model.convolModel(hpf, data, N)
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

        filtered_data = Model.convolModel(bpf, data, N)
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

    def __bsf():
        bsf = Processing.bsf(fc1_bsf, fc2_bsf, m_bsf, dt)
        tf_bsf = Analysis.transferFunction(bsf)
        deltaF_bsf = 2 * f_bound / (2 * m_bsf)

        filtered_data = Model.convolModel(bsf, data, N)
        Re, Im = Analysis.fourier(filtered_data, N)
        specter = Analysis.spectrFourier(Re, Im, N//2)

        plt.subplot(1, 3, 1)
        make_line_plot('Частотная характеристика фильтра bsf', np.arange(0, f_bound, deltaF_bsf),
                       tf_bsf, 'Freq [Hz]', 'Amplitude')
        plt.subplot(1, 3, 2)
        plt.ylim(-120, 120)
        make_line_plot('Данные после фильтра bsf', np.arange(N) * dt,
                       filtered_data, 'Time [index]', 'Amplitude')
        plt.subplot(1, 3, 3)
        plt.ylim(0, 40)
        make_line_plot('Спектр данных после фильтра bsf', np.arange(0, f_bound, deltaF),
                       specter, 'Freq [Hz]', '|X_n|')


    filename = './pgp_dt0005.dat'
    data = read_data(filename)
    N = len(data)
    dt = 0.0005
    f_bound = 1 / (2 * dt)
    deltaF = 2 * f_bound / N

    fc_lpf, fc_hpf, fc1_bpf, fc2_bpf, fc1_bsf, fc2_bsf = 15, 200, 15, 200, 15, 200
    m_lpf, m_hpf, m_bpf, m_bsf = 64, 64, 64, 64
    inner_funcs = [__lpf, __hpf, __bpf, __bsf]
    for i in range(len(inner_funcs)):
        inner_funcs[i]()
        if i != len(inner_funcs) - 1:
            plt.tight_layout()
            fig.savefig(f"./plots/{title}_{i+1}.png")
            current_title = fig._suptitle.get_text()
            fig = plt.figure(figsize=(14, 8))
            fig.suptitle(current_title)

def test_wavfile_methods():
    filename = './rickroll.wav'
    audio_data, rate, N_audio = InOut.readWAV(filename)

    duration = N_audio / rate
    print(f"Частота дискретизации: {rate} Гц, Длина сигнала: {N_audio} отсчётов, Длительность: {duration:.2f} сек")

    start_sample = rate
    end_sample = start_sample + rate*5
    audio_segment = audio_data[start_sample:end_sample]

    amplified_segment = audio_segment * 1.5
    InOut.writeWAV('./amplified_rickroll.wav', amplified_segment, rate)


    plt.subplot(1, 2, 1)
    plt.ylim(-15000, 15000)
    make_line_plot('Исходный фрагмент', np.arange(len(audio_segment)) / rate,
                   audio_segment, 'Time [sec]', 'Amplitude')
    plt.subplot(1, 2, 2)
    plt.ylim(-15000, 15000)
    make_line_plot('Усиленный фрагмент', np.arange(len(amplified_segment)) / rate,
                   amplified_segment, 'Time [sec]', 'Amplitude')


if __name__ == '__main__':
    sns.set_theme(style="whitegrid")
    if not os.path.exists("./plots"):
        os.makedirs("./plots")

    test_methods = {
        # test_plot_pgp_dt0005_data: 'pgp_dt0005 data plot',
        test_filters_methods: 'Data filtration',
        # test_wavfile_methods: 'Wav file read-write'
    }

    for test_method, title in test_methods.items():
        fig = plt.figure(figsize=(14, 8))
        fig.suptitle(title)
        test_method()
        plt.tight_layout()
        fig.savefig(f"./plots/{title}.png")
    plt.show()
