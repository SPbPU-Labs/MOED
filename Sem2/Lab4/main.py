import os
import seaborn as sns
from common.in_out import InOut
from common.processing import Processing


lab_folder_path = "./Lab4/"
images_folder = lab_folder_path + "images"
image_files = ["photo1.jpg", "photo2.jpg", "photo3.jpg", "photo4.jpg", "HollywoodLC.jpg"]
C_gamma = {
    image_files[0]: (255, 0.5),
    image_files[1]: (255, 0.5),
    image_files[2]: (255, 2),
    image_files[3]: (255, 0.5),
    image_files[4]: (255, 0.5),
}
C_log = {
    image_files[0]: 10,
    image_files[1]: 10,
    image_files[2]: 10,
    image_files[3]: 10,
    image_files[4]: 10,
}

grace_img = "grace.jpg"
xcrs_files = ["c12-85v.xcr", "u0_2048x2500.xcr"]


def to_negative_image():
    def jpg_file():
        image_data, M, N = InOut.readJPG(lab_folder_path + grace_img)
        print(f"Размер изображения {grace_img}: {M}x{N}")
        InOut.displayImage(image_data, f"Исходное изображение {grace_img}")

        neg_image_data = Processing.negative(image_data)
        InOut.displayImage(neg_image_data, f"Негативное изображение {grace_img}")
        InOut.saveJPG(f"{images_folder}/neg/grace_neg.jpg", neg_image_data)

    def xcr_file(path: str, file_size: tuple[int, int]):
        image, header = InOut.readXCR(path, file_size)
        grayscale_image = InOut.to_grayscale(image)
        M, N = image.shape
        print(f"Размер рентгеновского изображения {filename}: {M}x{N}")
        InOut.displayImage(grayscale_image, f"Рентгеновское изображение {filename}.xcr")
        InOut.saveJPG(f"{images_folder}/{filename}.jpg", grayscale_image)

        negative_xcr = Processing.negative(grayscale_image)
        InOut.displayImage(negative_xcr, f"Негатив рентгеновского изображения {filename}")
        InOut.saveJPG(f"{images_folder}/neg/{filename}_neg.jpg", negative_xcr)

    xcr_data = {
        "c12-85v": (lab_folder_path + xcrs_files[0], (1024, 1024)),
        "u0_2048x2500": (lab_folder_path + xcrs_files[1], (2048, 2500))
    }
    jpg_file()
    for filename, data in xcr_data.items():
        xcr_file(data[0], data[1])

def gamma_and_log_transform():
    def gamma_transform():
        transformed_image_data = Processing.gamma_transform(image_data, C_gamma[image][0], C_gamma[image][1])
        transformed_image_data = InOut.to_grayscale(transformed_image_data)
        InOut.displayImage(transformed_image_data, f"{name} {image}")
        InOut.saveJPG(f"{images_folder}/gamma/{image[:-4]}_gamma.jpg", transformed_image_data)

    def log_transform():
        transformed_image_data = Processing.log_transform(image_data, C_log[image])
        transformed_image_data = InOut.to_grayscale(transformed_image_data)
        InOut.displayImage(transformed_image_data, f"{name} {image}")
        InOut.saveJPG(f"{images_folder}/log/{image[:-4]}_log.jpg", transformed_image_data)


    transforms = {
        "Gamma transform": gamma_transform,
        "Log transform": log_transform,
    }
    for name, transform in transforms.items():
        for image in image_files:
            image_data, M, N = InOut.readJPG(lab_folder_path + image)
            print(f"Размер изображения {image}: {M}x{N}")
            InOut.displayImage(image_data, f"Исходное изображение {image}")
            transform()


if __name__ == '__main__':
    sns.set_theme(style="whitegrid")
    os.makedirs(images_folder+"/gamma", exist_ok=True)
    os.makedirs(images_folder+"/log", exist_ok=True)
    os.makedirs(images_folder+"/neg", exist_ok=True)

    test_methods = {
        to_negative_image: 'jpg and .xcr images negative',
        gamma_and_log_transform: 'Gamma and log transform jpg images'
    }

    for test_method, title in test_methods.items():
        print(f"\n----{title}-----\n")
        test_method()
