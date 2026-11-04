from abc import ABC, abstractmethod
import numpy as np


class Kernel(ABC):
    def __init__(self):
        pass


    @abstractmethod
    def __call__(self,r: np.ndarray) -> np.ndarray:
        pass

class GaussianKernel(Kernel):
    def __init__(self, sigma: float = 1):
        super().__init__()

        if sigma <=0:
            raise  ValueError("Отклонение должно быть больше 0")
        self.sigma = sigma

    def __call__(self, r: np.ndarray) -> np.ndarray:
        return np.where(r <= 1, np.exp(-2 * r ** 2), 0.0)

