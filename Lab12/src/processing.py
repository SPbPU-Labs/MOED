import numpy as np
from model import Model

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
        lpwm2 = np.concatenate((lpw[::-1], lpw[1:]))

        return lpw, lpwm2

    @staticmethod
    def hpf(fc, m, dt):
        lpw, lpw2 = Processing.lpf(fc, m, dt)
        hpw = np.array([1 - lpw2[m] if k == m else -lpw2[k] for k in range(len(lpw2))])
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
        bsw = np.array([1 + lpw1_2[m] - lpw2_2[m] if k == m else lpw1_2[k] - lpw2_2[k] for k in range(len(lpw1_2))])
        return bsw
