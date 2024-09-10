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
                print(time_str)
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



class TrendFuncs:
    @staticmethod
    def linear_func(t: np.ndarray, a: float, b: float) -> np.ndarray:
        """
        Линейная функция x(t) = at
        :param t: Массив времени
        :param a: Параметр a для функции тренда.
        :param b: Параметр b для функции тренда.
        :return: Массив точек линейной функции
        """
        return a * t + b

    @staticmethod
    def exp_func(t: np.ndarray, a: float, b: float) -> np.ndarray:
        """
        Экспоненциальная функция x(t) = exp(-a*t)
        :param t: Массив времени
        :param a: Параметр a для функции тренда.
        :param b: Параметр b для функции тренда.
        :return: Массив точек экспоненциальной функции
        """
        return np.exp(-a * t) + b
