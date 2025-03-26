import os
from common.in_out import InOut
from common.processing import Processing

images_src = "./common/images/"
images_folder = "./Lab6/images"
xcr_files = ["c12-85v.xcr", "u0_2048x2500.xcr"]
xcr_data = {
    "c12-85v.xcr": (images_src + xcr_files[0], (1024, 1024)),
    "u0_2048x2500.xcr": (images_src + xcr_files[1], (2048, 2500))
}

def apply_artifact_detection():
    for filename, data in xcr_data.items():
        image, header = InOut.readXCR(data[0], data[1])
        grayscale_image = InOut.to_grayscale(image)
        rotated_image = InOut.rotate_ccw(grayscale_image)

        print(f"Обнаружение артефактов в {filename}...")
        f0 = Processing.detect_artifacts(rotated_image, dy=50)
        print(f"Обнаружена доминирующая частота f0 = {f0:.3f}\n")

def apply_artifact_suppression():
    for filename, data in xcr_data.items():
        image, header = InOut.readXCR(data[0], data[1])
        grayscale_image = InOut.to_grayscale(image)
        rotated_image = InOut.rotate_ccw(grayscale_image)

        M, N = image.shape
        print(f"Размер рентгеновского изображения {filename}: {M}x{N}")
        InOut.displayImage(rotated_image, f"Исходное рентгеновское изображение {filename}")
        InOut.saveJPG(f"{images_folder}/{filename[:-4]}.jpg", rotated_image)

        print(f"Подавление артефактов в {filename}...")
        f0 = Processing.detect_artifacts(rotated_image, dy=50, show_plots=False)
        filtered_image = Processing.suppress_artifacts(rotated_image, f0)
        InOut.displayImage(filtered_image, f"Рентгеновское изображение с подавлением артефактов")
        InOut.saveJPG(f"{images_folder}/filtered_{filename[:-4]}.jpg", filtered_image)

if __name__ == '__main__':
    os.makedirs(images_folder, exist_ok=True)

    test_methods = {
        # apply_artifact_detection: "Detection of artifacts",
        apply_artifact_suppression: "Suppression of artifacts"
    }

    for test_method, title in test_methods.items():
        print(f"\n----{title}-----\n")
        test_method()
