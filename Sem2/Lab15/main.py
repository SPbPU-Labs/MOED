import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def debug_show(image, title="Debug"):
    """Функция для отладки: показывает изображение в уменьшенном размере"""
    plt.figure(figsize=(6, 4))
    plt.imshow(image, cmap="gray")
    plt.title(title)
    plt.show()
    cv2.imwrite(f"./Lab15/images/{title}.jpg", image)


def process_image(image_path, target_size):
    # Загрузка изображения
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Изображение не найдено или путь неверный")

    # Предварительная обработка
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    debug_show(gray, "Original Image")

    # Бинаризация
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    debug_show(binary, "Binary Image")

    # Поиск контуров
    contours_img = np.zeros_like(binary)
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(contours_img, contours, -1, (255, 255, 255), 1)
    debug_show(contours_img, "All Contours")

    # Вариант 1: камни, где оба размера равны S
    v1_count = 0
    v1_stones = []

    # Вариант 2: камни, где хотя бы один размер равен S, а другой меньше
    v2_count = 0
    v2_stones = []

    for cnt in contours:
        # Получаем ограничивающий прямоугольник
        _, _, w, h = cv2.boundingRect(cnt)

        # Вариант 1: оба размера равны S
        if target_size == w and target_size == h:
            v1_count += 1
            v1_stones.append(cnt)

        # Вариант 2: хотя бы один размер равен S, другой меньше
        if (target_size == w and h < target_size) or (
            target_size == h and w < target_size
        ):
            v2_count += 1
            v2_stones.append(cnt)

    print(f"Вариант 1: {v1_count} камней")
    print(f"Вариант 2: {v2_count} камней")

    mask_v1 = np.zeros_like(binary)
    mask_v2 = np.zeros_like(binary)

    # Визуализация результатов
    # Вариант 1 - зеленые контуры
    cv2.drawContours(mask_v1, v1_stones, -1, (255, 255, 255), 1)
    debug_show(mask_v1, "All Contours V1")

    # Вариант 2 - синие контуры
    cv2.drawContours(mask_v2, v2_stones, -1, (255, 255, 255), 1)
    debug_show(mask_v2, "All Contours V2")

    # Создаем красное подкрашивание
    blue_overlay = np.zeros_like(image)
    blue_overlay[:] = (255, 0, 0)  # Красный цвет в RGB

    # Применяем маску: только выделенные камни будут красными
    colored_stones_v1 = cv2.bitwise_and(blue_overlay, blue_overlay, mask=mask_v1)
    colored_stones_v2 = cv2.bitwise_and(blue_overlay, blue_overlay, mask=mask_v2)

    # Накладываем на оригинальное изображение
    result_img_v1 = cv2.addWeighted(image, 0.5, colored_stones_v1, 1, 0)
    result_img_v2 = cv2.addWeighted(image, 0.5, colored_stones_v2, 1, 0)

    # Добавляем текст с количеством
    cv2.putText(
        result_img_v1,
        f"Stones: {v1_count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2,
    )

    cv2.putText(
        result_img_v2,
        f"Stones: {v2_count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2,
    )

    debug_show(result_img_v1, "Result Image V1")
    debug_show(result_img_v2, "Result Image V2")

    return v1_count, v2_count, result_img_v1, result_img_v2


if __name__ == "__main__":
    os.makedirs("./Lab15/images", exist_ok=True)
    process_image("./common/images/stones.jpg", target_size=10)
