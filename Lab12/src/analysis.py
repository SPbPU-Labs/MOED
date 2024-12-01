import numpy as np
from enum import Enum

from src.model import Model
from src.processing import Processing


class ACFType(Enum):
    ACF = 1
    COV = 2

class CCFType(Enum):
    CCF = 1,
    CCOV = 2

class StatType(Enum):
    MIN = "min",
    MAX = "max",
    MEAN = "mean",
    VARIANCE = "variance",
    STD_DEV = "std_dev",
    SKEW = "skew",
    SKEW_RATIO = "skew_ratio",
    EXCESS = "excess",
    KURTOSIS = "kurtosis",
    MSE = "mse",
    RMSE = "rmse",
    ALL = "all"


class Analysis:
    @staticmethod
    def stationarity(data: np.ndarray, N: int, P: float) -> bool:
        """
        Определение стационарности вектора размерности MxN по среднему значению
        :param data: Вектор размерности MxN
        :param N: Кол-во столбцов вектора MxN
        :param P: Параметр стационарности
        :return: Возвращает true, если вектор стационарен, иначе возвращает false
        """
        # Вычисление средних значений по ансамблю
        means_val = np.mean(data, axis=0)
        for m in range(N):
            for n in range(m + 1, N):
                delta = abs(means_val[m] - means_val[n])
                if delta >= P:
                    return False
        return True

    @staticmethod
    def ergodicity(data: np.ndarray, M: int, N: int, P: float) -> bool:
        """
        Определение эргодичности вектора размерности MxN по среднему значению
        :param data: Вектор размерности MxN
        :param M: Кол-во строк вектора MxN
        :param N: Кол-во столбцов вектора MxN
        :param P: Параметр стационарности
        :return: Возвращает true, если вектор эргодичен, иначе возвращает false
        """
        # Вычисление средних значений по реализациям
        means_val = np.mean(data, axis=1)
        if not Analysis.stationarity(data, N, P):
            return False

        for m in range(M):
            for n in range(m + 1, M):
                delta = abs(means_val[m] - means_val[n])
                if delta >= P:
                    return False
        return True

    @staticmethod
    def hist(data, M) -> np.ndarray:
        """Расчет плотности вероятностей (гистограммы)"""
        hist, bin_edges = np.histogram(data, bins=M, density=True)
        return hist

    @staticmethod
    def acf(data, N, type=ACFType.ACF):
        """Расчет автокорреляционной или ковариационной функции на основе типа."""
        mean = np.mean(data)
        acf_values = []

        for L in range(N):
            numerator = np.sum((data[:N-L] - mean) * (data[L:] - mean))
            if type == ACFType.ACF:
                denominator = np.sum((data - mean) ** 2)
                acf_value = numerator / denominator
            elif type == ACFType.COV:
                acf_value = numerator / N
            else:
                raise ValueError("Unknown function type: expected ACFType.ACF or ACFType.COV")
            acf_values.append(acf_value)

        return np.array(acf_values)

    @staticmethod
    def ccf(dataX, dataY, N, type=CCFType.CCOV):
        """Расчет кросс-корреляционной функции для двух наборов данных."""
        mean_x = np.mean(dataX)
        mean_y = np.mean(dataY)
        ccf_values = []

        for L in range(N):
            numerator = np.sum((dataX[:N-L] - mean_x) * (dataY[L:] - mean_y))
            if type == CCFType.CCF:
                denominator = np.sum((dataX - mean_x) ** 2)
                ccf_value = numerator / denominator
            elif type == CCFType.CCOV:
                ccf_value = numerator / N
            else:
                raise ValueError("Unknown function type: expected CCFType.CCF or CCFType.CCOV")
            ccf_values.append(ccf_value)

        return np.array(ccf_values)

    @staticmethod
    def fourier(data, N):
        """
        Вычисляет дискретное преобразование Фурье для сигнала data.
        :param data: np.ndarray
            Входной сигнал (временной ряд).
        :param N: int
            Длина массива входного сигнала

        :return tuple: (Re, Im)
            Кортеж из вещественной и мнимой частей.
        """
        Re = np.zeros(N)
        Im = np.zeros(N)
        for n in range(N):
            for k in range(N):
                angle = 2 * np.pi * n * k / N
                Re[n] += data[k] * np.cos(angle)
                Im[n] += data[k] * np.sin(angle)
            Re[n] /= N
            Im[n] /= N
        return Re, Im

    @staticmethod
    def spectrFourier(Re, Im, N_half):
        """
        Вычисляет амплитудный спектр Фурье для заданного сигнала.
        """
        return np.sqrt(Re[:N_half]**2 + Im[:N_half]**2)

    @staticmethod
    def statistics(data, N, stat_type=StatType.ALL):
        """
        Вычисление статистических характеристик данных.
        :param data - входные данные
        :param N - размер данных
        :param type - тип статистики ('all' по умолчанию для расчета всех статистик)
        :return Возвращает словарь со значениями статистических характеристик.
        """

        def __min():
            return np.min(data)

        def __max():
            return np.max(data)

        def __mean():
            return np.mean(data)

        def __variance():
            return np.var(data)

        def __std_dev():
            return np.sqrt(__variance())

        def __skewness():
            mean = __mean()
            d3 = abs(data - mean)**3
            return sum(d3)/N

        def __skew_ratio():
            return __skewness()/(__std_dev()**3)

        def __excess():
            mean = __mean()
            d4 = abs(data - mean)**4
            return sum(d4)/N

        def __kurtosis():
            return __excess()/(__std_dev()**4) - 3

        def __mse():
            return np.mean(data ** 2)

        def __rmse():
            return np.sqrt(__mse())

        def __all():
            result = {}
            for key, val in switch_case.items():
                result[key] = val
            return result

        switch_case = {
            StatType.MIN: lambda: __min(),
            StatType.MAX: lambda: __max(),
            StatType.MEAN: lambda: __mean(),
            StatType.VARIANCE: lambda: __variance(),
            StatType.SKEW: lambda: __skewness(),
            StatType.SKEW_RATIO: lambda: __skew_ratio(),
            StatType.EXCESS: lambda: __excess(),
            StatType.KURTOSIS: lambda: __kurtosis(),
            StatType.MSE: lambda: __mse(),
            StatType.RMSE: lambda: __rmse(),
            StatType.ALL: lambda: __all()
        }

        return switch_case[stat_type]()

    @staticmethod
    def SNR(signal, noise):
        # Вычисляем стандартное отклонение сигнала и шума
        sigma_signal = np.std(signal)
        sigma_noise = np.std(noise)
        return 20 * np.log10(sigma_signal / sigma_noise)

    @staticmethod
    def transferFunction(data, m):
        N = len(data)
        Re, Im = Analysis.fourier(data, N)
        spectrum = Analysis.spectrFourier(Re, Im, N//2)
        return spectrum * (2*m+1)
