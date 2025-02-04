from typing import Callable
import numpy as np


class Model:
    def trend(self, func: Callable[[np.ndarray, float, float], np.ndarray], a: float, b: float, delta: float,
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

    def shift(self, data: np.ndarray, N: int, C: float) -> np.ndarray:
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

    def mult(self, data: np.ndarray, N: int, C: float) -> np.ndarray:
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
