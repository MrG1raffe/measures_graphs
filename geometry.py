from abc import ABC, abstractmethod
import numpy as np

class Geometry(ABC):
    @abstractmethod
    def distance(self, x: np.ndarray, y: np.ndarray) -> float:
        pass

    @abstractmethod
    def distance_matrix(self, coordinates: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def coordinates_to_pos(self, coordinates: np.ndarray) -> np.ndarray:
        """
        For visualization purposes.

        :return:
        """
        pass


class EuclideanGeometry(Geometry):
    def distance(self, x: np.ndarray, y: np.ndarray) -> float:
        return np.linalg.norm(x - y)

    def distance_matrix(self, coordinates: np.ndarray) -> np.ndarray:
        return np.sqrt(((coordinates.T[:, :, None] - coordinates.T[:, None, :])**2).sum(axis=0))

    def coordinates_to_pos(self, coordinates: np.ndarray) -> np.ndarray:
        return coordinates


class PoincareDisk(Geometry):
    def distance(self, x: np.ndarray, y: np.ndarray) -> float:
        return np.arccosh(1 + 2 * (x[0]**2 + y[0]**2 - 2 * x[0] * y[0] * np.cos(x[1] - y[1])) / (1 - x[0]**2) / (1 - y[0]**2))

    def distance_matrix(self, coordinates: np.ndarray) -> np.ndarray:
        r = coordinates[:, 0]
        phi = coordinates[:, 1]
        return np.arccosh(1 + 2 * (r[:, None]**2 + r[None, :]**2 - 2 * r[:, None] * r[None, :] *
                                   np.cos(phi[:, None] - phi[None, :])) / (1 - r[:, None]**2) / (1 - r[None, :]**2))

    def coordinates_to_pos(self, coordinates: np.ndarray) -> np.ndarray:
        pos = np.zeros_like(coordinates)
        pos[:, 0] = coordinates[:, 0] * np.cos(coordinates[:, 1])
        pos[:, 1] = coordinates[:, 0] * np.sin(coordinates[:, 1])
        return pos