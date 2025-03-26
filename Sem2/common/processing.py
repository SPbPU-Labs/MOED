from abc import ABC, abstractmethod

import numpy as np
from scipy.signal import hilbert
from scipy.fftpack import fft
from scipy.signal import correlate
import matplotlib.pyplot as plt


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

    @staticmethod
    def negative(image):
        """
        Преобразование изображения в негатив.

        :param image : np.ndarray: Исходное изображение.

        :return np.ndarray: Изображение в негативе.
        """
        L = np.max(image)
        return L - image

    @staticmethod
    def gamma_transform(image, C, gamma):
        """
        Гамма-преобразование изображения.
        s = C * r^γ

        :param image : np.ndarray: Исходное изображение.
        :param C : float: Коэффициент масштаба.
        :param gamma : float: Показатель степени.

        :return np.ndarray: Преобразованное изображение.
        """
        return np.array(C * (image / 255) ** gamma, dtype='uint8')

    @staticmethod
    def log_transform(image, C):
        """
        Логарифмическое преобразование изображения.
        s = C * lg(r+1)

        :param image : np.ndarray: Исходное изображение.
        :param C : float: Коэффициент масштаба.

        :return np.ndarray: Преобразованное изображение.
        """

        return np.array(C * np.log1p(image), dtype='uint8')

    @staticmethod
    def gradation_transform(image: np.ndarray) -> np.ndarray:
        """
        Градационное преобразование изображения

        Эквализация гистограммы для изображения по формуле:
            p(r_k) = n_k / (M * N)

        Функция распределения (CDF):
            C(r) = sum_{i=0}^{r} p(i)

        Пересчёт яркости по формуле:
            s = T(r) = L * C(r)
        где L — максимальное значение яркости исходного изображения.

        :param image : np.ndarray: Исходное изображение в оттенках серого размером MxN.

        :return np.ndarray: Изображение после эквализации гистограммы.
        """
        M, N = image.shape
        L = 255
        hist, bins = np.histogram(image.flatten(), bins=L+1, range=[0, L+1])
        total_pixels = M * N
        p = hist / total_pixels  # нормализованная гистограмма p(r_k)

        cdf = np.cumsum(p)  # накопленная функцию распределения (CDF)

        equalization_map = np.floor(L * cdf).astype(np.uint8)  # отображающая функиця s
        equalized_image = equalization_map[image]
        return equalized_image

    @staticmethod
    def detect_artifacts(image: np.ndarray, dy: int = 50, show_plots: bool = True) -> float:
        """
        Анализ изображения для обнаружения артефактов.

        1. Вычисление амплитудных спектров:
            - Исходной строки.
            - Производной строки.
            - Автокорреляционной функции (АКФ) производной строки.
            - Взаимной корреляции (ВКФ) между производными двух строк.
        2. Вычисление спектров взаимной корреляции (ВКФ) между производными двух строк.
        3. Поиск доминирующих максимумов спектров АКФ и ВКФ в диапазоне частот [0.25 – 0.5].
        4. Вычисление средней частоты совпадающих масимумов f0.

        Args:
            image (np.ndarray): двумерное рентгеновское изображение.
            dy (int): шаг по вертикали.
            show_plots (bool): флаг отображения графиков спектров.

        Returns:
            float: Средняя частота доминирующего максимума f0.
        """
        def __compute_spectrum(signal: np.ndarray) -> np.ndarray:
            """
            Амплитудный спектр Фурье сигнала.

            Args:
                signal (np.ndarray): одномерный массив данных (строка изображения).

            Returns:
                np.ndarray: амплитудный спектр.
            """
            N = len(signal)
            spectrum = np.abs(fft(signal)) / N
            return spectrum[:N // 2]  # Возвращаем только положительные частоты

        def __compute_autocorrelation(signal: np.ndarray) -> np.ndarray:
            """
            АКФ (автокорреляционная функция) для строки изображения.

            Args:
                signal (np.ndarray): одномерный массив данных.

            Returns:
                 np.ndarray: автокорреляционная функция.
            """
            return correlate(signal, signal, mode='full')[len(signal)-1:]

        def __compute_crosscorrelation(signal1: np.ndarray, signal2: np.ndarray) -> np.ndarray:
            """
            ВКФ (взаимная корреляция) двух строк изображения.

            Args:
                signal1 (np.ndarray): первая строка изображения.
                signal2 (np.ndarray): вторая строка изображения.

            Returns:
                np.ndarray: взаимная корреляция.
            """
            return correlate(signal1, signal2, mode='full')[len(signal1)-1:]

        height, width = image.shape
        matched_freqs = []

        for y in range(0, height - dy, dy):
            row1 = image[y, :]
            row2 = image[y + dy, :]
            row1_derivative = np.gradient(row1)
            row2_derivative = np.gradient(row2)

            # Вычисление спектров
            spectrum_row1 = __compute_spectrum(row1)
            spectrum_deriv = __compute_spectrum(row1_derivative)
            spectrum_acf = __compute_spectrum(__compute_autocorrelation(row1_derivative))
            spectrum_vcf = __compute_spectrum(__compute_crosscorrelation(row1_derivative, row2_derivative))

            # Построение графиков спектров
            if show_plots and y == 0:
                plt.figure(figsize=(12, 6))

                plt.subplot(2, 2, 1)
                plt.plot(spectrum_row1)
                plt.title(f"Спектр строки {y}")
                plt.xlabel("Частота")
                plt.ylabel("Амплитуда")

                plt.subplot(2, 2, 2)
                plt.plot(spectrum_deriv, color="red")
                plt.title(f"Спектр производной строки {y}")
                plt.xlabel("Частота")
                plt.ylabel("Амплитуда")

                plt.subplot(2, 2, 3)
                plt.plot(spectrum_acf, color="green")
                plt.title(f"Спектр АКФ производной строки {y}")
                plt.xlabel("Частота")
                plt.ylabel("Амплитуда")

                plt.subplot(2, 2, 4)
                plt.plot(spectrum_vcf, color="purple")
                plt.title(f"Спектр ВКФ между строками {y} и {y+dy}")
                plt.xlabel("Частота")
                plt.ylabel("Амплитуда")

                plt.tight_layout()
                plt.show()

            # Создаём частотную шкалу
            freqs = np.linspace(0, 0.5, len(spectrum_acf))

            # Ищем пики в АКФ в диапазоне [0.25; 0.5]
            peak_acf_indices = np.where((freqs >= 0.25) & (freqs <= 0.5))[0]
            peak_acf_freq = freqs[peak_acf_indices[np.argmax(spectrum_acf[peak_acf_indices])]]

            # Ищем пики в ВКФ в диапазоне [0.25; 0.5]
            peak_vcf_indices = np.where((freqs >= 0.25) & (freqs <= 0.5))[0]
            peak_vcf_freq = freqs[peak_vcf_indices[np.argmax(spectrum_vcf[peak_vcf_indices])]]

            # Проверяем совпадение пиков
            if abs(peak_acf_freq - peak_vcf_freq) < 0.02:  # Допускаем небольшую разницу
                matched_freqs.append((peak_acf_freq + peak_vcf_freq) / 2)  # Среднее по двум спектрам

        # Возвращаем среднее арифметическое найденных частот
        return np.mean(matched_freqs) if matched_freqs else 0.0

    @staticmethod
    def suppress_artifacts(image: np.ndarray, f0: float, show_plots: bool = True) -> np.ndarray:
        """
        Подавление артефактов на рентгеновском изображении с помощью режекторного фильтра.

        Args:
            image (np.ndarray): Исходное изображение.
            f0 (float): Частота артефакта.
            show_plots (bool): отображение графика АЧХ фильтра

        Returns:
            np.ndarray: Обработанное изображение.
        """
        def __potter_filter(signal: np.ndarray, f0: float, m: int = 32) -> np.ndarray:
            """
            Режекторный фильтр Поттера.

            Args:
                signal : np.ndarray: входной сигнал (строка изображения).
                f0 : float: частота артефакта.
                m : int: размерность фильтра.

            Returns:
                np.ndarray: Отфильтрованный сигнал.
            """
            dt = 1  # Шаг дискретизации
            fc1, fc2 = f0 - 0.05, f0 + 0.05  # Полоса заграждения

            # Формируем фильтр
            lpw = np.sinc(2 * fc1 * (np.arange(-m, m + 1) * dt)) - np.sinc(2 * fc2 * (np.arange(-m, m + 1) * dt))
            lpw /= np.sum(lpw)  # Нормализация

            if show_plots:
                spectrum = np.abs(np.fft.fft(lpw, 512))
                freqs = np.linspace(0, 0.5, len(spectrum) // 2)

                plt.figure(figsize=(8, 4))
                plt.plot(freqs, spectrum[:len(freqs)])
                plt.title(f"АЧХ режекторного фильтра (f0 = {f0:.3f})")
                plt.xlabel("Частота")
                plt.ylabel("Амплитуда")
                plt.grid()
                plt.show()

            # Применяем свёртку
            return np.convolve(signal, lpw, mode='same')

        height, width = image.shape
        filtered_image = np.copy(image)

        # Фильтрация по строкам
        for y in range(height):
            filtered_image[y, :] = __potter_filter(image[y, :], f0)
            show_plots = False

        # Фильтрация по столбцам
        for x in range(width):
            filtered_image[:, x] = __potter_filter(filtered_image[:, x], f0)

        return filtered_image


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
