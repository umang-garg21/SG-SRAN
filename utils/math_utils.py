import numpy as np


def safe_norm(
    x: np.ndarray,
    axis: int = 0,
    keepdims: bool = True,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Compute vector norm with epsilon guard to avoid division by zero.

    Parameters
    ----------
    x : np.ndarray
        Input array.
    axis : int, default=-1
        Axis to compute norm over.
    keepdims : bool, default=True
        Whether to keep dimensions (for safe broadcasting).
    eps : float, default=1e-12
        Minimum threshold for norm.

    Returns
    -------
    np.ndarray
        Norm array (broadcastable to input).
    """
    norms = np.linalg.norm(x, axis=axis, keepdims=keepdims)
    np.maximum(norms, eps, out=norms)  # in-place clamp for speed
    return norms
