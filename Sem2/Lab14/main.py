import os
import matplotlib.pyplot as plt

from utils import read_bin_image
from MRIImageOptimizer import EqualizationOptimizer, CLAHEOptimizer, SobelOptimizer


def process_image(filename: str, MRIImageOptimizer=EqualizationOptimizer):
    """Полная обработка одного изображения"""
    size = 256 if "256" in filename else 512
    image = read_bin_image(filename, size, size)

    equilized, adjusted = MRIImageOptimizer.optimize(image)

    plt.figure(figsize=(12, 8))
    plt.suptitle(f"MRI optimizer: {MRIImageOptimizer.__name__}")

    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap="gray")
    plt.title(f"Original: {file}")

    plt.subplot(1, 3, 2)
    plt.imshow(equilized, cmap="gray")
    plt.title(f"Step 1: {file}")

    plt.subplot(1, 3, 3)
    plt.imshow(adjusted, cmap="gray")
    plt.title(f"Step 2: {file}")

    plt.tight_layout()
    plt.show()

    return adjusted


# Пример использования
if __name__ == "__main__":
    os.makedirs("./Lab14/images", exist_ok=True)
    
    files = [
        "spine-V_x512.bin",
        "spine-H_x256.bin",
        "brain-V_x256.bin",
        "brain-H_x512.bin",
    ]

    for file in files:
        for optimizer in [
            EqualizationOptimizer,
            CLAHEOptimizer,
            SobelOptimizer,
        ]:
            print(f"Processing {file}...")
            filePath = f"./common/images/{file}"
            adjusted_image = process_image(filePath, optimizer)
            adjusted_image.tofile("./Lab14/images/" + optimizer.__name__ + "_" + file)
