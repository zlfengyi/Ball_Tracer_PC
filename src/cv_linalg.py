from __future__ import annotations

import cv2
import numpy as np


def _as_f64(value: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(value, dtype=np.float64)


def projection_matrix(
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """Build a 3x4 camera projection matrix without NumPy matmul."""
    Rt = np.concatenate([_as_f64(R), _as_f64(t).reshape(3, 1)], axis=1)
    return cv2.gemm(_as_f64(K), Rt, 1.0, None, 0.0)


def matvec(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Multiply matrix and vector without NumPy matmul."""
    col = _as_f64(vector).reshape(-1, 1)
    return cv2.gemm(_as_f64(matrix), col, 1.0, None, 0.0).reshape(-1)


def smallest_right_singular_vector(matrix: np.ndarray) -> np.ndarray:
    """Return the right-singular vector for the smallest singular value."""
    _, _, vt = cv2.SVDecomp(_as_f64(matrix))
    return vt[-1]


def solve_least_squares(matrix: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Solve min ||Ax - b||_2 with OpenCV's SVD solver."""
    ok, result = cv2.solve(
        _as_f64(matrix),
        _as_f64(values).reshape(-1, 1),
        flags=cv2.DECOMP_SVD,
    )
    if not ok:
        raise RuntimeError("least-squares solve failed")
    return result.reshape(-1)
