import numpy as np
from scipy.spatial import Delaunay


class Hull1D:
    def __init__(self, points: np.ndarray):
        self.max = np.max(points)
        self.min = np.min(points)

    def find_simplex(self, xi: np.ndarray) -> float:
        if self.min <= xi <= self.max:
            return 1.0
        return -1.0


def in_hull(p: np.ndarray, hull: Delaunay | Hull1D):
    return hull.find_simplex(p) >= 0
