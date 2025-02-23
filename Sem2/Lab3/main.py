import os
import seaborn as sns
from common.in_out import InOut
from common.processing import Processing

grace_path = "./Lab3/grace.jpg"
grace_folder_path = "./Lab3/images/grace"

xcr1_path = "./Lab3/c12-85v.xcr"
xcr2_path = "./Lab3/u0_2048x2500.xcr"
xcr_folder_path = "./Lab3/images/xcr"

grace_incr_ratio = 1.3
grace_decr_ratio = 0.7
c12_85v_xcr_ratio = 0.6
u0_2048x2500_xcr_ratio = 0.24


def resize_grace_by_nearest_neighbor():
    image_data, M, N = InOut.readJPG(grace_path)
    print(f"Размер исходного изображения grace: {M}x{N}")
    InOut.displayImage(image_data, "Оригинальное изображение")
    InOut.saveJPG(f"{grace_folder_path}/grace_original.jpg", image_data)

    incr_image_data = Processing.nearest_neighbor_resize(image_data, grace_incr_ratio)
    incr_M, incr_N = incr_image_data.shape
    print(f"Размер увеличенного (х{grace_incr_ratio}) изображения grace: {incr_M}x{incr_N}")
    InOut.displayImage(incr_image_data, f"Увеличинное изображение (x{grace_incr_ratio})")
    InOut.saveJPG(f"{grace_folder_path}/grace_incr_neighbor.jpg", incr_image_data)

    decr_image_data = Processing.nearest_neighbor_resize(image_data, grace_decr_ratio)
    decr_M, decr_N = decr_image_data.shape
    print(f"Размер уменьшенного (x{grace_decr_ratio}) изображения grace: {decr_M}x{decr_N}")
    InOut.displayImage(image_data, f"Уменьшенное изображение (x{grace_decr_ratio})")
    InOut.saveJPG(f"{grace_folder_path}/grace_decr_neighbor.jpg", decr_image_data)

def resize_grace_by_bilinear_interpolation():
    image_data, M, N = InOut.readJPG(grace_path)
    print(f"Размер исходного изображения grace: {M}x{N}")
    InOut.displayImage(image_data, "Оригинальное изображение")
    InOut.saveJPG(f"{grace_folder_path}/grace_original.jpg", image_data)

    incr_image_data = Processing.bilinear_resize(image_data, 1.3)
    incr_M, incr_N = incr_image_data.shape
    print(f"Размер увеличенного (х{grace_incr_ratio}) изображения grace: {incr_M}x{incr_N}")
    InOut.displayImage(incr_image_data, f"Увеличинное изображение (x{grace_incr_ratio})")
    InOut.saveJPG(f"{grace_folder_path}/grace_incr_bilinear.jpg", incr_image_data)

    decr_image_data = Processing.bilinear_resize(image_data, 0.7)
    decr_M, decr_N = decr_image_data.shape
    print(f"Размер уменьшенного (x{grace_decr_ratio}) изображения grace: {decr_M}x{decr_N}")
    InOut.displayImage(image_data, f"Уменьшенное изображение (x{grace_decr_ratio})")
    InOut.saveJPG(f"{grace_folder_path}/grace_decr_bilinear.jpg", decr_image_data)

def rotate_and_resize_xcr():
    def resize(path: str, ratio: float, file_size: tuple[int, int]):
        image, header = InOut.readXCR(path, file_size)
        grayscale_image = InOut.to_grayscale(image)
        M, N = image.shape
        print(f"Размер исходного рентгеновского изображения {filename}: {M}x{N}")
        InOut.displayImage(grayscale_image, f"Рентгеновское изображение {filename}.xcr")
        InOut.saveJPG(f"{xcr_folder_path}/{filename}.jpg", grayscale_image)

        resized_xcr = method(InOut.rotate_ccw(grayscale_image), ratio)
        resized_M, resized_N = resized_xcr.shape
        print(f"Размер изменённого (х{ratio}) рентгеновского изображения {filename}: {resized_M}x{resized_N}")
        InOut.displayImage(resized_xcr, f"Изменённое рентгеновское изображение {filename} (x{ratio})")
        InOut.saveJPG(f"{xcr_folder_path}/{filename}_{method_name}.jpg", resized_xcr)

    xcr_data = {
        "c12-85v": (xcr1_path, c12_85v_xcr_ratio, (1024, 1024)),
        "u0_2048x2500": (xcr2_path, u0_2048x2500_xcr_ratio, (2048, 2500))
    }
    methods = {
        "neighbor": Processing.nearest_neighbor_resize,
        "bilinear": Processing.bilinear_resize
    }
    for method_name, method in methods.items():
        for filename, data in xcr_data.items():
            print(f"###File: {filename}, Method: {method_name}###")
            resize(data[0], data[1], data[2])


if __name__ == '__main__':
    sns.set_theme(style="whitegrid")
    os.makedirs(grace_folder_path, exist_ok=True)
    os.makedirs(xcr_folder_path, exist_ok=True)

    test_methods = {
        resize_grace_by_nearest_neighbor: 'Grace image resize by nearest neighbor method',
        resize_grace_by_bilinear_interpolation: 'Grace image resize by bilinear method',
        rotate_and_resize_xcr: '.xcr images rotate and resize by methods',
    }

    for test_method, title in test_methods.items():
        print(f"\n----{title}-----\n")
        test_method()
