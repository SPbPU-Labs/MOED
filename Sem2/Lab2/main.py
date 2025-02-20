import os
import seaborn as sns
from common.in_out import InOut
from common.processing import Processing

filename = "./Lab2/c12-85v.xcr"
c_shift = 30
c_mult = 1.3


def read_xcr():
    image, header = InOut.readXCR(filename)
    grayscale_image = InOut.to_grayscale(image)
    InOut.displayImage(grayscale_image, "Рентгеновское изображение")
    InOut.saveJPG(f"./Lab2/images/xcr.jpg", grayscale_image)

def rotate_xcr():
    image, header = InOut.readXCR(filename)
    grayscale_image = InOut.to_grayscale(image)
    rotated_image = InOut.rotate_ccw(grayscale_image)
    InOut.displayImage(rotated_image, "Рентгеновское изображение повёрнутое на 90°")
    InOut.saveJPG(f"./Lab2/images/xcr_rotated.jpg", rotated_image)

def save_xcr():
    image, header = InOut.readXCR(filename)
    grayscale_image = InOut.to_grayscale(image)
    rotated_image = InOut.rotate_ccw(grayscale_image)
    InOut.save_xcr(f"./Lab2/xcr_rotated_s.xcr", rotated_image, header)
    InOut.save_bin(f"./Lab2/xcr_rotated.bin", rotated_image)


if __name__ == '__main__':
    sns.set_theme(style="whitegrid")
    if not os.path.exists("./Lab2/images"):
        os.makedirs("./Lab2/images")

    test_methods = {
        # read_xcr: '.xcr image read',
        # rotate_xcr: '.xcr image rotate',
        save_xcr: '.xcr image save',
    }

    for test_method, title in test_methods.items():
        test_method()
