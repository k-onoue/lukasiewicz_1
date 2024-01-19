import numpy as np
from scipy.linalg import polar

def get_nearest_psd(A: np.ndarray) -> np.ndarray:
    B = .5 * (A.T + A)
    _, H = polar(B)
    return (B + H) / 2