import numpy as np
from numpy import float64
from numpy.typing import NDArray, Float64Like  # type: ignore[attr-defined]

from typing import cast


def _as_1d(
    x: Float64Like | NDArray[np.float64], m: int, name: str
) -> NDArray[np.float64]:
    if isinstance(x, (float, int, np.floating, np.integer)):
        return np.full((m,), float64(x), dtype=float64)
    arr = np.asarray(x, dtype=float64).reshape(-1)
    if arr.shape != (m,):
        raise ValueError(f"{name} must be scalar or shape ({m},), got {arr.shape}.")
    return arr


def _sym(A: NDArray[np.float64]) -> NDArray[np.float64]:
    return (A + A.T) * 0.5


def _apply_floor_diag(R: NDArray[np.float64], floor: float) -> NDArray[np.float64]:
    R = R.copy()
    diag = np.diag(R)
    diag = np.maximum(diag, float64(floor))
    np.fill_diagonal(R, diag)
    return R


def _compose_cov_from_std_and_corr(
    std: NDArray[np.float64],
    corr: NDArray[np.float64] | None,
) -> NDArray[np.float64]:
    # diagonal-only if no corr
    if corr is None:
        return np.diag(std * std).astype(float64)

    corr = _sym(np.asarray(corr, dtype=float64))
    np.fill_diagonal(corr, 1.0)
    # covariance = D * Corr * D
    D = np.diag(std.astype(float64))
    return (D @ corr @ D).astype(float64)


def _safe_chol_pd(R: NDArray[np.float64], jitter: float = 0.0) -> NDArray[np.float64]:
    """
    Not strictly required for build_R, but handy:
    checks PD and can add jitter if user passes weird corr matrices.
    """
    R = _sym(np.asarray(R, dtype=float64))
    try:
        np.linalg.cholesky(R)
        return R
    except np.linalg.LinAlgError:
        if jitter <= 0:
            return R  # let KalmanFilter decide what to do
        return R + float64(jitter) * np.eye(R.shape[0], dtype=float64)


def _hp_noise_std(y: NDArray[np.float64], lamb: float) -> NDArray[np.float64]:
    """
    Compute per-series std of HP residual: y - trend.
    This is a heuristic for measurement noise magnitude.

    Implemented with a standard HP smoothing linear system:
      trend = argmin_t sum (y - trend)^2 + lamb * sum (Δ² trend)^2
    """
    y = np.asarray(y, dtype=float64)
    T, m = y.shape
    if T < 4:
        # not enough points for second-difference penalty to be meaningful
        return cast(np.ndarray, np.std(y, axis=0, ddof=0).astype(float64))

    # Build (T,T) banded system: (I + lamb * D'D) trend = y
    # where D is (T-2, T) second-difference operator.
    I = np.eye(T, dtype=float64)

    D = np.zeros((T - 2, T), dtype=float64)
    for i in range(T - 2):
        D[i, i] = 1.0
        D[i, i + 1] = -2.0
        D[i, i + 2] = 1.0

    A = I + float64(lamb) * (D.T @ D)  # (T,T)

    # Solve for each series
    trend = np.empty_like(y, dtype=float64)

    for j in range(m):
        trend[:, j] = np.linalg.solve(A, y[:, j])

    resid = y - trend
    std = np.std(resid, axis=0, ddof=0).astype(float64)
    # avoid zeros
    return cast(np.ndarray, np.maximum(std, float64(1e-12)).astype(float64))
