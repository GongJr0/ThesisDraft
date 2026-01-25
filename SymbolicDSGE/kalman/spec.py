from dataclasses import dataclass
import numpy as np
from numpy import float64
from numpy.typing import NDArray
from typing import Literal
from abc import ABC

CorrLike = float | int | np.floating | NDArray[np.float64]
StdLike = float | int | np.floating | NDArray[np.float64]


class RSpec(ABC):
    @staticmethod
    def corr_to_mat(corr: CorrLike | None, m: int) -> NDArray[np.float64] | None:
        if corr is None:
            return None

        if isinstance(corr, (float, int, np.floating)):
            rho = float(corr)
            mat = np.full((m, m), rho, dtype=float64)
            np.fill_diagonal(mat, 1.0)
            return mat

        mat = np.asarray(corr, dtype=float64)
        if mat.shape != (m, m):
            raise ValueError(f"corr matrix must have shape ({m}, {m}), got {mat.shape}")

        # symmetry + diag checks (tolerances can be parameters if you want)
        if not np.allclose(mat, mat.T, atol=1e-10, rtol=0.0):
            raise ValueError("corr matrix must be symmetric")
        if not np.allclose(np.diag(mat), 1.0, atol=1e-8, rtol=0.0):
            raise ValueError("corr matrix diagonal must be 1.0")

        return mat


@dataclass
class RConst(RSpec):
    std: StdLike
    corr: CorrLike | None = None
    floor: float = 1e-12

    def __post_init__(self) -> None:
        # Keep std scalar canonicalization only
        if isinstance(self.std, (float, int, np.floating)):
            self.std = float64(self.std)
        else:
            self.std = np.asarray(self.std, dtype=float64)


@dataclass
class RMoment(RSpec):
    mode: Literal["innov", "resid"] = "innov"
    init: RSpec | None = None
    burn_in: int = 0
    shrink: float = 0.0
    floor: float = 1e-12
    corr: CorrLike | None = None


@dataclass
class RDiagEM(RSpec):
    init: RSpec
    n_iter: int = 10
    floor: float = 1e-12
    damp: float = 1.0


@dataclass
class RHP(RSpec):
    lamb: float = 1600.0
    corr: CorrLike | None = None
    floor: float = 1e-12
