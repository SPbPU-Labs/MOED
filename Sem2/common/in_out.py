import matplotlib.pyplot as plt
from scipy.io import wavfile
from PIL import Image
import numpy as np


class InOut:
    @staticmethod
    def readWAV(filename):
        rate, data = wavfile.read(filename)
        N = len(data)
        return data, rate, N

    @staticmethod
    def writeWAV(filename, data, rate):
        wavfile.write(filename, rate, data)
        
    @staticmethod
    def readJPG(filename):
        image = Image.open(filename).convert("L")
        data = np.array(image)
        M, N = data.shape
        return data, M, N
    
    @staticmethod
    def displayImage(data, title="Изображение"):
        plt.figure(figsize=(6, 6))
        if len(data.shape) == 3:  # Цветное изображение (RGB)
            plt.imshow(data)
        else:
            plt.imshow(data, cmap="gray")
        plt.title(title)
        plt.axis("off")
        plt.show()

    @staticmethod
    def saveJPG(filename, data):
        image = Image.fromarray(np.clip(data, 0, 255).astype("uint8"))
        image.save(filename)
        print(f"Изображение сохранено как {filename}")

    @staticmethod
    def to_grayscale(image) -> np.ndarray:
        """
        Приведение изображение к шкале серости (диапазону 0-255).

        :param image : np.ndarray Исходный массив изображения.
        :return np.ndarray Нормализованный массив с диапазоном значений 0-255
        """
        x_min, x_max = image.min(), image.max()
        normalized_image = ((image - x_min) / (x_max - x_min)) * 255
        return normalized_image.astype(np.uint8)

    @staticmethod
    def readXCR(filename):
        """
        Чтение изображения с расширением файла .xcr.

        :param
        filename : str
            Путь к файлу.

        :returns
        tuple : (image, header)
            Изображение (1024x1024) и заголовок (2048 байт).
        """
        with open(filename, "rb") as f:
            header = f.read(2048)
            raw_data = np.fromfile(f, dtype=np.uint16, count=1024 * 1024)

        # Перестановка байтов (Big-endian → Little-endian)
        image = np.frombuffer(raw_data.byteswap(), dtype=np.uint16).reshape(1024, 1024)
        return image, header

    @staticmethod
    def rotate_ccw(image):
        """Поворот изображения на 90° против часовой стрелки"""
        return np.rot90(image)

    @staticmethod
    def save_bin(filename, image):
        """
        Сохранение изображения в бинарный файл .bin

        :param filename : str Имя выходного файла.
        :param image : np.ndarray Данные изображения.
        """
        image.astype(np.uint16).tofile(filename)
        print(f"Изображение сохранено как {filename}")

    @staticmethod
    def save_xcr(filename, image, header):
        """
        Сохранение изображения в файл .xcr

        :param filename : str Имя выходного файла.
        :param image : np.ndarray Данные изображения.
        :param header : bytes Оригинальный заголовок.
        """
        with open(filename, "wb") as f:
            f.write(header)  # Запись заголовка
            f.write(image.astype(np.uint16).byteswap().tobytes())  # Запись данных с перестановкой байтов
        print(f"Изображение сохранено как {filename}")
