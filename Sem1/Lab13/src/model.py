import hashlib
import time
from typing import Callable
import numpy as np


class Model:
    @staticmethod
    def trend(func: Callable[[np.ndarray, float, float], np.ndarray], a: float, b: float, delta: float,
              N: int) -> np.ndarray:
        """
        Генерация трендовые данные на основе функции.
        :param func: Функция для генерации данных тренда. Принимает параметры a, t (массив времени) и b, возвращает массив данных.
        :param a: Параметр a для функции тренда.
        :param b: Параметр b для функции тренда.
        :param delta: Шаг времени.
        :param N: Количество элементов в массиве данных.
        :return: Массив данных тренда numpy.
        """
        t = np.arange(0, N * delta, delta)
        return func(t, a, b)

    @staticmethod
    def shift(data: np.ndarray, N: int, C: float) -> np.ndarray:
        """
        Смещение данных на значение С.
        :param data: Данные для смещения (массив numpy).
        :param N: Количество элементов массива данных.
        :param C: Значение смещения.
        :return: Массив данных numpy смещённый на значение С.
        :raises ValueError: Если размер массива данных != N.
        """
        if len(data) != N:
            raise ValueError("Размерность массива не соответствует N")
        return data + C

    @staticmethod
    def mult(data: np.ndarray, N: int, C: float) -> np.ndarray:
        """
        Умножение данных на значение С
        :param data: Данные для умножения (массив numpy).
        :param N: Количество элементов массива данных.
        :param C: Значение умножения.
        :return: Массив данных numpy умноженный на значение С.
        :raises ValueError: Если размер массива данных != N.
        """
        if len(data) != N:
            raise ValueError("Размерность массива не соответствует N")
        return data * C

    @staticmethod
    def noise(N: int, R: int) -> np.ndarray:
        """
        Расчёт случайного шума в диапазоне значений [-R,R]
        с использованием всроенного генератора псевдослучайных чисел numpy
        :param N: Длина данных
        :param R: Значение диапазона
        :return: Массив numpy случайного шума
        """
        rand_array = np.random.rand(N)
        return Model.__x_k(rand_array, R)

    @staticmethod
    def myNoise(N: int, R: int) -> np.ndarray:
        """
        Расчёт случайного шума в диапазоне значений [-R,R]
        с использованием простого самописного генератора псевдослучайных чисел
        :param N: Длина данных
        :param R: Значение диапазона
        :return: Массив numpy случайного шума
        """
        def rand() -> np.ndarray:
            x = np.zeros(N)
            for i in range(N):
                time_str = str(time.perf_counter()).encode('utf-8')
                hash_md5 = hashlib.md5(time_str).hexdigest()
                hash_int = int(hash_md5, 16)
                x[i] = hash_int / 2**128  # [0, 1]
            return x

        rand_array = rand()
        return Model.__x_k(rand_array, R)

    @staticmethod
    def __x_k(t: np.ndarray, R: int) -> np.ndarray:
        x_min = t.min()
        x_max = t.max()
        return ((t-x_min) / (x_max - x_min) - 0.5) * 2 * R

    @staticmethod
    def spikes(N: int, M: int, R: float, Rs: float) -> np.ndarray:
        """
        Генерирует импульсный шум на интервале длиной N
        :param N: Длина данных
        :param M: Диапазон случайно генерируемого количества выбросов M·[N·0.1%, N·1%]
        :param R: Опорное значение амплитуды выбросов
        :param Rs: Диапазон изменения амплитуд выбросов [R ± Rs, R ± Rs]
        :return: Массив numpy импульсного шума
        """
        data = np.zeros(N)

        min_spikes_count = int(N * 0.001)
        max_spikes_count = int(N * 0.01)
        spikes_count = np.random.randint(min_spikes_count * M, max_spikes_count * M + 1)

        spike_indices = np.random.choice(np.arange(N), spikes_count, replace=False)
        spike_amplitudes = np.random.uniform(R - Rs, R + Rs, spikes_count)

        data[spike_indices] = spike_amplitudes * np.random.choice([-1, 1], size=spikes_count)
        return data

    @staticmethod
    def spikes2(N, positions, amplitudes):
        x = np.zeros(N)
        for pos, amp in zip(positions, amplitudes):
            x[pos] = amp
        return x

    @staticmethod
    def randVector(M: int, N: int, R: int, noise_method: Callable[[int, int], np.ndarray]) -> np.ndarray:
        """
        Генерация случайного вектор размерности MxN, состоящего из М реализаций
        случайного шума длины N каждая в диапазоне R
        :param M: Кол-во строк вектора
        :param N: Кол-во столбцов вектора
        :param R: Диапазон случайно генерируемых значений
        :param noise_method: Функция генерации N случайных значений в диапазоне R
        :return: Вектор размерности MxN
        """
        vectors = []
        for _ in range(M):
            vector = noise_method(N, R)
            vectors.append(vector)
        return np.array(vectors)

    @staticmethod
    def harm(N = 1000, A_0 = 100.0, f_0 = 15, dt = 0.001) -> np.ndarray:
        """
        Гармонический процесс
        x(t) = {x_k} = A_0 * sin(2π*f_0*Δt*k), k=0,1,2,...,N-1
        """
        k = np.arange(N)
        x = A_0 * np.sin(2 * np.pi * f_0 * dt * k)
        return x

    @staticmethod
    def polyHarm(N = 100, A = [100, 15, 20], f = [33, 5, 170], M = 3, dt = 0.002) -> np.ndarray:
        """
        Полигармонический процесс
        x(t) = {x_k} = ∑(A_i*sin(2π*f_0*Δt*k)), k=0,1,2,...,N-1
        """
        k = np.arange(N)
        x = np.zeros(N)
        for i in range(len(k)):
            summ = 0.0
            for j in range(M):
                summ += A[j] * np.sin(2 * np.pi * f[j] * dt * k[i])
            x[i] = summ
        return x

    @staticmethod
    def addModel(data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
        """Аддитивная модель"""
        return data1 + data2

    @staticmethod
    def multModel(data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
        """Мультипликативная модель"""
        return data1 * data2

    @staticmethod
    def convolModel(x, h, N):
        """Дискретная свёртка с отбрасыванием последних M значений"""
        y = np.convolve(x, h, mode='full')
        return y[:N]

    @staticmethod
    def heartRate(N=1000, M=200, A=1, f=7, dt=0.005, a=30, b=1):
        # Импульсная реакция модели сердечной мышцы
        h1 = Model.harm(M, A, f, dt)
        h2 = Model.trend(TrendFuncs.exp_func, a, b, dt, M)
        h = Model.multModel(h1, h2)
        h = 120 * h / np.max(h)

        # Управляющая функция ритма
        positions = [200, 400, 600, 800]
        amplitudes = np.random.uniform(0.9, 1.1, len(positions))
        x = Model.spikes2(N, positions, amplitudes)

        # Свёртка
        y = Model.convolModel(x, h, M)
        noise = np.random.normal(0, 1, len(y))
        y = Model.addModel(y, noise)
        return h, x, y

    @staticmethod
    def badHeartRate(N=1000, M=200, A=1, f=7, dt=0.005, a=30, b=1):
        # Импульсная реакция модели сердечной мышцы
        h1 = Model.harm(M, A, f, dt)
        h2 = Model.trend(TrendFuncs.exp_func, a, b, dt, M)
        h = Model.multModel(h1, h2)
        h = 120 * h / np.max(h)

        # Управляющая функция ритма
        x = Model.spikes(N, 1, 1, 0.3)

        # Свёртка
        y = Model.convolModel(x, h, M)
        noise = np.random.normal(0, 1, len(y))
        y = Model.addModel(y, noise)
        return h, x, y


class TrendFuncs:
    @staticmethod
    def linear_func(t: np.ndarray, a: float, b: float) -> np.ndarray:
        """
        Линейная функция x(t) = at+b
        :param t: Массив времени
        :param a: Параметр a для функции тренда.
        :param b: Параметр b для функции тренда.
        :return: Массив точек линейной функции
        """
        return a * t + b

    @staticmethod
    def exp_func(t: np.ndarray, a: float, b: float) -> np.ndarray:
        """
        Экспоненциальная функция x(t) = exp(-a*t)*b
        :param t: Массив времени
        :param a: Параметр a для функции тренда.
        :param b: Параметр b для функции тренда.
        :return: Массив точек экспоненциальной функции
        """
        return np.exp(-a * t) * b
