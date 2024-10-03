import numpy as np


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
