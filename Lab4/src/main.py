from src.analysis import Analysis
from src.model import Model


N = [100, 1000, 1000]
M = [100, 100, 1000]
R = 100
P = [R * 0.01, R * 0.03, R * 0.05]

# ANSI-коды для цветов
RED = '\033[31m'
GREEN = '\033[32m'
YELLOW = '\033[33m'
BLUE = '\033[34m'
RESET = '\033[0m'


if __name__ == '__main__':

    for noise_method in [Model.noise, Model.myNoise]:
        for i in range(3):
            data = Model.randVector(M[i], N[i], R, noise_method)
            stat = Analysis.stationarity(data, N[i], P[i])
            ergo = Analysis.ergodicity(data, M[i], N[i], P[i])
            print(f"Входные параметры:{YELLOW} M = {M[i]}, N = {N[i]}, R = {R}, P = {P[i]}{RESET}")
            if noise_method is Model.noise:
                print(f"Метод генерации случайных данных:{BLUE} noise(){RESET}")
            else:
                print(f"Метод генерации случайных данных:{BLUE} myNoise(){RESET}")
            print(f"Полученные результаты: {GREEN}Стационарность{RESET} – {stat}; {GREEN}Эргодичность{RESET} – {ergo}\n")

    # i = 0
    # while i <= 10000:
    #     i += 1
    #     P = 22
    #     data = Model.randVector(M[0], N[0], R, Model.noise)
    #     stat = Analysis.stationarity(data, N[0], P)
    #     ergo = Analysis.ergodicity(data, M[0], N[0], P)
    #
    #     if ergo:
    #         print(f"Входные параметры:{YELLOW} M = {M[0]}, N = {N[0]}, R = {R}, P = {P}{RESET}")
    #         print(f"Метод генерации случайных данных:{BLUE} noise(){RESET}")
    #         print(f"Полученные результаты: {GREEN}Стационарность{RESET} – {stat}; {GREEN}Эргодичность{RESET} – {ergo}")
    #         print(f"i = {i}\n")

    print("Конец тестирования")
