import os

import matplotlib.pyplot as plt
import numpy as np

from common.analysis import Analysis
from common.in_out import InOut
from common.model import Model
from common.utils import make_line_plot

images_src = "./common/images/"
images_folder = "./Lab8/images"
test_image = "grace.jpg"


def apply_inverse_fourier_transform():
    N = 1000
    A0 = 100
    f = 25
    dt = 0.001
    time = np.arange(0, N * dt, dt)

    data = Model.harm(N, A0, f, dt)
    Re, Im = Analysis.fourier(data, N)

    restored_data = Analysis.inverseFourier(Re+Im, N)

    fig = plt.figure(figsize=(14, 8))
    fig.suptitle(title)
    plt.subplot(1, 2, 1)
    make_line_plot('Harm', time, data, 'Time [sec]', 'Amplitude')
    plt.subplot(1, 2, 2)
    make_line_plot('Inverse Fourier Harm', time, restored_data, 'Time [sec]', 'Amplitude')
    plt.tight_layout()
    fig.savefig(f"{images_folder}/{title}.png")
    plt.show()

def apply_image_fourier_transform():
    white_rect_image = np.zeros((256, 256))
    white_rect_image[100:150, 100:150] = 255  # Белый квадрат на чёрном фоне
    grace_image, _, _ = InOut.readJPG(images_src + test_image)

    images_data = {
        "White rect": white_rect_image,
        "Grace": grace_image,
    }
    for name, data in images_data.items():
        spectrum = Analysis.fourier2D(data)
        fig = plt.figure(figsize=(6, 6))
        plt.imshow(np.log1p(np.abs(np.fft.fftshift(spectrum))), cmap='gray')
        plt.title("Fourier spectrum")
        fig.savefig(f"{images_folder}/spectrum_{name}.png")
        plt.colorbar()

        restored_image = Analysis.inverseFourier2D(spectrum)

        fig = plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(data, cmap='gray')
        plt.title("Original Image")

        plt.subplot(1, 2, 2)
        plt.imshow(restored_image, cmap='gray')
        plt.title("Restored Image")
        fig.savefig(f"{images_folder}/compare_{name}.png")
    plt.show()


if __name__ == '__main__':
    os.makedirs(images_folder, exist_ok=True)

    test_methods = {
        apply_inverse_fourier_transform: "Inverse Fourier transform test",
        apply_image_fourier_transform: "Image fourier transform"
    }

    for test_method, title in test_methods.items():
        print(f"\n----{title}-----\n")
        test_method()
