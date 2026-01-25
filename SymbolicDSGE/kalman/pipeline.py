import numpy as np
from numpy import float64
from numpy.typing import NDArray

from .spec import *
from ._utils import (
    _as_1d,
    _compose_cov_from_std_and_corr,
    _apply_floor_diag,
    _safe_chol_pd,
    _hp_noise_std,
)


class KalmanPipeline:

    @staticmethod
    def build_R(
        y: NDArray[float64], spec: RSpec, *, pd_jitter: float = 0.0
    ) -> NDArray[float64]:
        """
        Build an (m,m) measurement noise covariance R from raw observations y.

        Important:
          - RMoment(mode="innov"/"resid") requires innovation/residual series.
            So this function uses RMoment.init (or raises) unless you call the
            dedicated helpers build_R_from_innov / build_R_from_resid.
        """

        y = np.asarray(y, dtype=float64)
        if y.ndim != 2:
            raise ValueError(f"y must be 2D (T,m) array, got shape {y.shape}.")
        _, m = y.shape

        if isinstance(spec, RConst):
            std = _as_1d(spec.std, m, "RConst.std")
            corr = spec.corr_to_mat(spec.corr, m)
            R = _compose_cov_from_std_and_corr(std, corr)
            R = _apply_floor_diag(R, spec.floor)
            return _safe_chol_pd(R, pd_jitter)

        elif isinstance(spec, RHP):
            # use raw y -> estimate noise std
            std = _hp_noise_std(y, spec.lamb)
            corr = spec.corr_to_mat(spec.corr, m)
            R = _compose_cov_from_std_and_corr(std, corr)
            R = _apply_floor_diag(R, spec.floor)
            return _safe_chol_pd(R, pd_jitter)

        elif isinstance(spec, RMoment):
            if spec.mode not in ("innov", "resid"):
                raise ValueError(f"Invalid RMoment.mode={spec.mode!r}.")
            if spec.init is None:
                raise ValueError(
                    "RMoment requires a bootstrap R via RMoment.init (e.g., RConst or RHP) "
                    "for the first Kalman pass."
                )
            return KalmanPipeline.build_R(y, spec.init, pd_jitter=pd_jitter)

        elif isinstance(spec, RDiagEM):
            raise ValueError(
                "RDiagEM cannot be built from y alone. "
                "Run a Kalman pass and call an EM updater that uses (innov, S, K, etc.)."
            )

        raise TypeError(f"Unsupported RSpec type: {type(spec)!r}")

    @staticmethod
    def build_R_from_moment(
        moment: NDArray[float64],
        spec: RMoment,
        *,
        pd_jitter: float = 0.0,
    ) -> NDArray[float64]:
        """
        Generic moment estimator. `moment` is either innovations v_t or residuals e_t.
        """

        M = np.asarray(moment, dtype=float64)
        if M.ndim != 2:
            raise ValueError(f"moment must be 2D (T,m) array, got shape {M.shape}.")

        T, m = M.shape
        burn = int(spec.burn_in)
        if (burn < 0) or (burn >= T):
            raise ValueError(f"Burn-in must be in [0, {T-1} (T-1)], got: {burn}.")
        M2 = M[burn:, :]

        var = np.var(M2, axis=0, ddof=0).astype(float64)
        shrink = float64(spec.shrink)

        if (shrink < 0.0) or (shrink > 1.0):
            raise ValueError(f"Shrinkage must be in [0.0, 1.0], got: {shrink}.")

        if shrink > 0.0:
            mu = float64(np.mean(var))
            var = (1 - shrink) * var + shrink * mu

        std = np.sqrt(np.maximum(var, float64(spec.floor)))

        corr = spec.corr_to_mat(spec.corr, m)
        R = _compose_cov_from_std_and_corr(std, corr)
        return _safe_chol_pd(R, pd_jitter)

    @staticmethod
    def build_R_from_innov(
        innov: NDArray[float64],
        spec: RMoment,
        *,
        pd_jitter: float = 0.0,
    ) -> NDArray[float64]:
        if spec.mode != "innov":
            raise ValueError(
                f"RMoment.mode={spec.mode!r} is incompatible with innovations. "
                "Use RMoment(mode='innov') or call build_R_from_resid(...)."
            )
        return KalmanPipeline.build_R_from_moment(innov, spec, pd_jitter=pd_jitter)

    @staticmethod
    def build_R_from_resid(
        resid: NDArray[float64],
        spec: RMoment,
        *,
        pd_jitter: float = 0.0,
    ) -> NDArray[float64]:
        if spec.mode != "resid":
            raise ValueError(
                f"RMoment.mode={spec.mode!r} is incompatible with residuals. "
                "Use RMoment(mode='resid') or call build_R_from_innov(...)."
            )
        return KalmanPipeline.build_R_from_moment(resid, spec, pd_jitter=pd_jitter)
