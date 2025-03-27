import os
from common.in_out import InOut
from common.model import Model, ImageNoiseLevel
from common.processing import Processing

images_src = "./common/images/"
images_folder = "./Lab7/images"
test_image = "MODELimage.jpg"

noise_methods = {
    "random": Model.image_random_noise,
    "impulse": Model.image_impulse_noise,
    "mixed": Model.image_mixed_noise,
}
noise_levels = [e for e in ImageNoiseLevel]

filter_methods = {
    "mean": Processing.mean_filter,
    "median": Processing.median_filtering,
}
filter_kernel_size = 3

def apply_image_noise_methods():
    image_data, M, N = InOut.readJPG(images_src + test_image)
    print(f"Размер изображения {test_image}: {M}x{N}")
    # InOut.displayImage(image_data, f"Исходное изображение {test_image}")
    InOut.saveJPG(f"{images_folder}/{test_image}", image_data)

    for method_name, method in noise_methods.items():
        for level1 in noise_levels:
            if method is Model.image_mixed_noise:
                for level2 in noise_levels:
                    noised_image = method(image_data, level1, level2)
                    # InOut.displayImage(noised_image, f"Зашумленное изображение; method: {method_name}, "
                    #                    f"level (rand/imp): {level1.name}/{level2.name}")
                    InOut.saveJPG(f"{images_folder}/noised/{method_name}_{level1.name}_{level2.name}_{test_image}",
                                  noised_image)
            else:
                noised_image = method(image_data, level1)
                # InOut.displayImage(noised_image, f"Зашумленное изображение; method: {method_name}, level: {level1.name}")
                InOut.saveJPG(f"{images_folder}/noised/{method_name}_{level1.name}_{test_image}", noised_image)

def apply_image_noise_filtration():
    image_data, M, N = InOut.readJPG(images_src + test_image)
    print(f"Размер изображения {test_image}: {M}x{N}")
    # InOut.displayImage(image_data, f"Исходное изображение {test_image}")
    InOut.saveJPG(f"{images_folder}/{test_image}", image_data)

    for filter_name, filter_method in filter_methods.items():
        for noise_name, noise_method in noise_methods.items():
            for level1 in noise_levels:
                if noise_method is Model.image_mixed_noise:
                    for level2 in noise_levels:
                        noised_image = noise_method(image_data, level1, level2)
                        # InOut.displayImage(noised_image, f"Зашумленное изображение; method: {noise_name}, "
                        #                                  f"level (rand/imp): {level1.name}/{level2.name}")
                        denoised_image = filter_method(noised_image, filter_kernel_size)
                        # InOut.displayImage(denoised_image, f"Фильтрованное изображение; method: {filter_name}, "
                        #                                  f"kernel_size : {filter_kernel_size}")
                        InOut.saveJPG(f"{images_folder}/filtered/{filter_name}_{noise_name}_{filter_kernel_size}_{level1.name}_{level2.name}_{test_image}",
                                      denoised_image)
                else:
                    noised_image = noise_method(image_data, level1)
                    # InOut.displayImage(noised_image, f"Зашумленное изображение; method: {noise_name}, level: {level1.name}")
                    denoised_image = filter_method(noised_image, filter_kernel_size)
                    # InOut.displayImage(denoised_image, f"Фильтрованное изображение; method: {filter_name}, "
                    #                                  f"kernel_size : {filter_kernel_size}")
                    InOut.saveJPG(f"{images_folder}/filtered/{filter_name}_{noise_name}_{filter_kernel_size}_{level1.name}_{test_image}",
                                  denoised_image)


if __name__ == '__main__':
    os.makedirs(images_folder + "/noised", exist_ok=True)
    os.makedirs(images_folder + "/filtered", exist_ok=True)

    test_methods = {
        # apply_image_noise_methods: "Image noise methods apply",
        apply_image_noise_filtration: "Image noise filtration"
    }

    for test_method, title in test_methods.items():
        print(f"\n----{title}-----\n")
        test_method()
