from abc import ABC, abstractmethod

from scipy.signal import hilbert
import numpy as np


class Processing:
    @staticmethod
    def antiSpike(data, N, R):
        """
        Функция обнаружения и удаления выбросов за пределами диапазона R.
        Заменяет выбросы линейной интерполяцией.
        """
        processed_data = np.copy(data)
        spikes = np.abs(data) > R
        for i in range(1, N - 1):
            if spikes[i]:
                processed_data[i] = (data[i - 1] + data[i + 1]) / 2
        return processed_data

    @staticmethod
    def antiTrendLinear(data, N):
        """
        Функция удаления линейного тренда с помощью первой производной.
        Возвращает данные с удалённым линейным трендом.
        """
        return np.diff(data, append=data[0])

    @staticmethod
    def antiTrendNonLinear(data, N, W):
        """
        Функция удаления нелинейного тренда методом скользящего среднего.
        Возвращает данные с удалённым нелинейным трендом.
        """
        # Скользящее среднее для устранения тренда
        trend = np.convolve(data, np.ones(W)/W, mode='valid')

        # Для выравнивания размера массивов дополним trend до размера N
        trend_padded = np.concatenate((trend, np.zeros(N - len(trend))))

        # Убираем тренд из данных
        detrended_data = data - trend_padded
        return detrended_data

    @staticmethod
    def antiShift(data):
        mean_value = np.mean(data)
        proc_data = data - mean_value
        return proc_data

    @staticmethod
    def antiNoise(data, M, N):
        result = []
        for i in range(N):
            avg = 0
            for j in range(M):
                avg = avg + data[j][i]
            avg = avg / M
            result.append(avg)
        return result

    @staticmethod
    def lpf(fc, m, dt):
        d = [0.35577019, 0.2436983, 0.07211497, 0.00630165]
        lpw = np.zeros(m + 1)
        fact = 2 * fc * dt
        lpw[0] = fact
        arg = fact * np.pi

        for i in range (1, m + 1):
            lpw[i] = np.sin(arg * i) / (np.pi * i)
        lpw[m] /= 2

        sumg = lpw[0]
        for i in range (1, m + 1):
            sum_ = d[0]
            arg = np.pi * i / m
            for j in range (1, 4):
                sum_ += 2 * d[j] * np.cos(arg * j)
            lpw[i] *= sum_
            sumg += 2 * lpw[i]

        lpw /= sumg

        # Симметризация весов
        lpwm2 = np.concatenate((lpw[-1::-1], lpw[1:]))

        return lpw, lpwm2

    @staticmethod
    def hpf(fc, m, dt):
        lpw, lpw2 = Processing.lpf(fc, m, dt)
        hpw = np.array([1 - lpw2[m] if k == m else -lpw2[k] for k in range(2 * m + 1)])
        return hpw

    @staticmethod
    def bpf(fc1, fc2, m, dt):
        lpw1, lpw1_2 = Processing.lpf(fc1, m, dt)
        lpw2, lpw2_2 = Processing.lpf(fc2, m, dt)
        bpw = lpw2_2 - lpw1_2
        return bpw

    @staticmethod
    def bsf(fc1, fc2, m, dt):
        lpw1, lpw1_2 = Processing.lpf(fc1, m, dt)
        lpw2, lpw2_2 = Processing.lpf(fc2, m, dt)
        bsw = np.array([1 + lpw1_2[m] - lpw2_2[m] if k == m else lpw1_2[k] - lpw2_2[k] for k in range(2 * m + 1)])
        return bsw

    @staticmethod
    def shift2D(data2D, C):
        """
        Смещение данных двумерного массива на значение C.

        :param data: Двумерный массив данных для смещения.
        :param C: Значение смещения.
        :return: Двумерный массив данных, смещённый на значение C.
        """
        return np.clip(data2D + C, 0, 255)

    @staticmethod
    def mult2D(data2D, C):
        """
        Мультипликативная модель для 2D массивов

        :param data2D: 2D массив данных для умножения.
        :param C: Значение умножения.
        :return: 2D массив данных, умноженный на значение C.
        """
        return np.clip(data2D * C, 0, 255)

    @staticmethod
    def nearest_neighbor_resize(image, scale):
        """
        Масштабирование изображения методом ближайшего соседа.

        :param image : np.ndarray: Исходное изображение.
        :param scale : float: Коэффициент масштабирования.

        :return np.ndarray: Масштабированное изображение.
        """
        M, N = image.shape[:2]  # Размеры оригинала
        new_M, new_N = int(M * scale), int(N * scale)  # Новые размеры

        # Массив с размером (new_M, new_N, 2) для grayscale или 3 для RGB
        resized_image = np.zeros((new_M, new_N, *image.shape[2:]), dtype=image.dtype)

        for i in range(new_M):
            for j in range(new_N):
                src_x = min(int(i / scale), M - 1)
                src_y = min(int(j / scale), N - 1)
                resized_image[i, j] = image[src_x, src_y]

        return resized_image

    @staticmethod
    def bilinear_resize(image, scale):
        """
        Масштабирование изображения методом билинейной интерполяции.

        :param image : np.ndarray: Исходное изображение.
        :param scale : float: Коэффициент масштабирования.

        :return np.ndarray: Масштабированное изображение.
        """
        M, N = image.shape[:2]
        new_M, new_N = int(M * scale), int(N * scale)
        resized_image = np.zeros((new_M, new_N, *image.shape[2:]), dtype=image.dtype)

        for i in range(new_M):
            for j in range(new_N):
                x = i / scale
                y = j / scale

                x0, x1 = int(x), min(int(x) + 1, M - 1)
                y0, y1 = int(y), min(int(y) + 1, N - 1)

                a = x - x0
                b = y - y0

                resized_image[i, j] = (1 - a) * (1 - b) * image[x0, y0] + \
                                      (1 - a) * b * image[x0, y1] + \
                                      a * (1 - b) * image[x1, y0] + \
                                      a * b * image[x1, y1]

        return resized_image.astype(image.dtype)


class Modulator(ABC):
    @abstractmethod
    def modulate(self, N: int, A_c: float, f_c: float, f_m: float, mod_index: float, dt: float) -> np.ndarray:
        """
        Модуляция сообщения.

        :param N: Длина модулирующего сигнала
        :param A_c: Амплитуда модулируемого сигнала
        :param f_c: Несущая частота
        :param f_m: Модулирующая частота
        :param mod_index: Индекс модуляции
        :param dt: Частота дискретизации
        :return: Частотно-модулированный сигнал
        """
        pass

    @abstractmethod
    def demodulate(self, modulated_signal: np.ndarray, f_c: float, f_m: float, sampling_rate: float) -> np.ndarray:
        """
        Демодуляция сообщения.

        :param modulated_signal: FM-сигнал
        :param f_c: Частота несущей
        :param f_m: Модулирующая частота
        :param sampling_rate: Частота дискретизации 1/dt
        :return: Демодулированный сигнал
        """
        pass

class FMModulator(Modulator):
    def modulate(self, N: int, A_c: float, f_c: float, f_m: float, mod_index: float, dt: float) -> np.ndarray:
        """
        Модуляция сообщения методом частотной модуляции.
        """
        t = np.arange(N) * dt
        return A_c * np.cos(2 * np.pi * f_c * t + mod_index * np.sin(2 * np.pi * f_m * t))

    def demodulate(self, modulated_signal: np.ndarray, f_c: float, f_m: float, sampling_rate: float) -> np.ndarray:
        """
        Демодуляция FM-сигнала.
        """
        from model import Model

        analytic_signal = hilbert(modulated_signal)
        instant_phase = np.unwrap(np.angle(analytic_signal))
        diff = np.gradient(instant_phase)

        m = 32
        lpf, lpf2 = Processing.lpf(f_m, m, 1/sampling_rate)
        filtered_data = Model.convolModel(lpf2, diff, len(diff))
        Processing.antiShift(filtered_data)
        return filtered_data
