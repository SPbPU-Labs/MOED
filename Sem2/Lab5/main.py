import os
import seaborn as sns

from common.analysis import Analysis
from common.in_out import InOut
from common.processing import Processing


images_folder = "./Lab5/images"
images_src = "./common/images/"
modified_images_src = "./Lab5/modified_images/"
jpg_files = ["HollywoodLC.jpg", "photo1.jpg", "photo2.jpg", "photo3.jpg", "photo4.jpg", "grace.jpg"]
xcrs_files = ["c12-85v.xcr", "u0_2048x2500.xcr"]


def apply_gradation_transform():
    def jpg_files_apply():
        for image in jpg_files:
            image_data, M, N = InOut.readJPG(images_src + image)
            print(f"Размер изображения {image}: {M}x{N}")
            InOut.displayImage(image_data, f"Исходное изображение {image}")

            grad_transform_data = Processing.gradation_transform(image_data)
            InOut.displayImage(grad_transform_data, f"Градационное преобразование изображения {image}")
            InOut.saveJPG(f"{images_folder}/grad_trans/{image}", grad_transform_data)

    def xcr_files_apply(path: str, file_size: tuple[int, int]):
        image, header = InOut.readXCR(path, file_size)
        grayscale_image = InOut.to_grayscale(image)
        image_data = InOut.rotate_ccw(grayscale_image)
        M, N = image.shape
        print(f"Размер рентгеновского изображения {filename}: {M}x{N}")
        InOut.displayImage(image_data, f"Исходное рентгеновское изображение {filename}.xcr")

        grad_transform_data = Processing.gradation_transform(image_data)
        InOut.displayImage(grad_transform_data,
                           f"Градационное преобразование рентгеновского изображения {filename}")
        InOut.saveJPG(f"{images_folder}/grad_trans/{filename}.jpg", grad_transform_data)


    xcr_data = {
            "c12-85v": (images_src + xcrs_files[0], (1024, 1024)),
            "u0_2048x2500": (images_src + xcrs_files[1], (2048, 2500))
        }
    jpg_files_apply()
    for filename, data in xcr_data.items():
        xcr_files_apply(data[0], data[1])

def apply_images_compare():
    images = ["HollywoodLC.jpg", "photo1.jpg", "photo2.jpg", "photo3.jpg", "photo4.jpg", "grace.jpg",
             "c12-85v.xcr", "u0_2048x2500.xcr"]
    xcr_data = {
        "c12-85v.xcr": (images_src + xcrs_files[0], (1024, 1024)),
        "u0_2048x2500.xcr": (images_src + xcrs_files[1], (2048, 2500))
    }
    for folder in os.listdir(modified_images_src):
        for modified_image in os.listdir(modified_images_src + folder):
            for image in images:
                if image[:-4] in modified_image:
                    if image.endswith(".jpg"):
                        image_data, M, N = InOut.readJPG(images_src + image)
                    elif image.endswith(".xcr"):
                        image_data, header = InOut.readXCR(xcr_data[image][0], xcr_data[image][1])
                        image_data = InOut.to_grayscale(image_data)
                        image_data = InOut.rotate_ccw(image_data)
                        M, N = image_data.shape
                    else:
                        break
                    print(f"Размер изображения {image}: {M}x{N}")
                    InOut.displayImage(image_data, f"Исходное изображение {image}")

                    modified_image_path = f"{modified_images_src}{folder}/{modified_image}"
                    modified_image_data, M_m, N_m = InOut.readJPG(modified_image_path)
                    print(f"Размер модифицированного изображения {modified_image}: {M_m}x{N_m}")
                    InOut.displayImage(modified_image_data, f"Модифицированное изображение {modified_image}")

                    compare_data = Analysis.compare_images(image_data, modified_image_data)
                    InOut.displayImage(compare_data, f"Разностное изображение {image}")
                    os.makedirs(f"{images_folder}/compare/{folder}", exist_ok=True)
                    InOut.saveJPG(f"{images_folder}/compare/{folder}/{modified_image}", compare_data)


if __name__ == '__main__':
    sns.set_theme(style="whitegrid")
    os.makedirs(images_folder+"/grad_trans", exist_ok=True)
    os.makedirs(images_folder+"/compare", exist_ok=True)

    test_methods = {
        apply_gradation_transform: 'jpg and .xcr images gradation transform',
        apply_images_compare: 'Modified and original images compare'
    }

    for test_method, title in test_methods.items():
        print(f"\n----{title}-----\n")
        test_method()
