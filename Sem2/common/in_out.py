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
        plt.imshow(data, cmap="gray")
        plt.title(title)
        plt.axis("off")
        plt.show()

    @staticmethod
    def saveJPG(filename, data):
        image = Image.fromarray(np.clip(data, 0, 255).astype("uint8"))
        image.save(filename)
        print(f"Изображение сохранено как {filename}")
    