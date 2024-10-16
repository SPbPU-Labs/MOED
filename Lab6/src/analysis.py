import numpy as np
from enum import Enum

class ACFType(Enum):
    ACF = 1
    COV = 2

class CCFType(Enum):
    CCF = 1,
    CCOV = 2

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
