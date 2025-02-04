import os
import seaborn as sns
from common.in_out import InOut
from common.processing import Processing

filename = "./Lab1/grace.jpg"
c_shift = 30
c_mult = 1.3


def read_img():
    image_data, M, N = InOut.readJPG(filename)
    print(f"Размер изображения: {M}x{N}")
    InOut.displayImage(image_data, "Оригинальное изображение")
    InOut.saveJPG(f"./Lab1/images/original.jpg", image_data)

def shift_img():
    image_data, M, N = InOut.readJPG(filename)
    image_data = Processing.shift2D(image_data, c_shift)
    InOut.displayImage(image_data, "Сдвинутое изображение")
    InOut.saveJPG(f"./Lab1/images/shifted.jpg", image_data)

def mult_img():
    image_data, M, N = InOut.readJPG(filename)
    image_data = Processing.mult2D(image_data, c_mult)
    InOut.displayImage(image_data, "Увеличенное изображение")
    InOut.saveJPG(f"./Lab1/images/mult.jpg", image_data)
    

if __name__ == '__main__':
    sns.set_theme(style="whitegrid")
    if not os.path.exists("./Lab1/images"):
        os.makedirs("./Lab1/images")

    test_methods = {
        read_img: 'Image read',
        shift_img: 'Image shift',
        mult_img: 'Image mult',
    }

    for test_method, title in test_methods.items():
        test_method()
