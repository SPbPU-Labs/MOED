import numpy as np


def read_bin_image(filename, width, height):
    """Чтение 16-битного бинарного изображения"""
    with open(filename, "rb") as f:
        data = np.fromfile(f, dtype=np.int16)
    return data.reshape((height, width))


def normalize_image(image):
    """Нормализация изображения к диапазону [0, 1]"""
    return (image - np.min(image)) / (np.max(image) - np.min(image))


def gradation_transform(image: np.ndarray, L: int) -> np.ndarray:
    """Градационное преобразование изображений"""
    # Нормализуем изображение перед обработкой
    image_norm = normalize_image(image)

    # Вычисляем гистограмму (используем 256 бинов для упрощения)
    hist, _ = np.histogram(image_norm.flatten(), bins=256, range=[0, 1])
    p = hist / image.size  # нормализованная гистограмма

    cdf = np.cumsum(p)  # CDF

    # Создаем маппинг значений с сохранением исходного диапазона
    equalization_map = np.floor(L * cdf).astype(np.int16)

    # Масштабируем входные значения к диапазону [0, 255] для маппинга
    scaled_image = ((image_norm * 255).astype(np.uint8)).flatten()
    equalized_image = equalization_map[scaled_image].reshape(image.shape)

    return equalized_image


def percentile_stretch(image, low_p, high_p, L):
    """Перцентильное растяжение изображения"""
    low, high = np.percentile(image, (low_p, high_p))
    adjusted = np.clip((image - low) * (L / (high - low)), 0, L)
    return adjusted

def sobel_filter(image, flag):
    """Фильтрация изображения с помощью фильтра Собеля."""
    image_1 = np.array(image)
    image_2 = np.array(image)
    if flag == 1:
        sobel_x = np.array([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]])

        sobel_y = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])
    if flag == 2:
        sobel_x = np.array([[0, 1, 2],
                            [-1, 0, 1],
                            [-2, -1, 0]])

        sobel_y = np.array([[-2, -1, 0],
                            [-1, 0, 1],
                            [0, 1, 2]])

    lx = image.shape[0]
    ly = image.shape[1]
    mx = 1
    my = 1

    for i in range(mx, lx - mx):
        for e in range(my, ly - my):
            le = i - mx
            lr = i + mx + 1
            re = e - my
            rr = e + my + 1
            gx = np.sum(image_1[le:lr, re:rr] * sobel_x)
            gy = np.sum(image_1[le:lr, re:rr] * sobel_y)
            image_2[i, e] = np.sqrt(gx ** 2 + gy ** 2)
    return image_2
