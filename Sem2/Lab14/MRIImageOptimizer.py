from abc import ABC, abstractmethod

import cv2
import numpy as np

from utils import gradation_transform, percentile_stretch, sobel_filter

class MRIImageOptimizer(ABC):
    @abstractmethod
    def optimize(self, image: np.ndarray) -> np.ndarray:
        """Автоматическая настройка яркости и контраста"""
        pass


class EqualizationOptimizer(MRIImageOptimizer):
    @staticmethod
    def optimize(image: np.ndarray) -> np.ndarray:
        equalized = gradation_transform(image, 65535)

        adjusted = percentile_stretch(equalized, 1, 100, 65535)
        return equalized.astype(np.int16), adjusted.astype(np.int16)
    
class CLAHEOptimizer(MRIImageOptimizer):
    @staticmethod
    def optimize(image: np.ndarray) -> np.ndarray:
        image_norm = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized = clahe.apply(image_norm)

        adjusted = percentile_stretch(equalized, 0, 99, 255)
        return equalized.astype(np.int16), adjusted.astype(np.int16)
    
class SobelOptimizer(MRIImageOptimizer):
    @staticmethod
    def optimize(image: np.ndarray) -> np.ndarray:
        sobel = sobel_filter(image, 1)
        image = image + sobel
        
        hist = np.bincount(image.ravel(), minlength=np.max(image)+1)
        data_hist = hist / (image.shape[0] * image.shape[1])
        data_cdf = np.cumsum(data_hist)
        maxval = np.max(image)
        adjusted = np.round(data_cdf[image] * maxval).astype(image.dtype)
        return image, adjusted
