from scipy.stats import norm, t, uniform
from scipy.stats._distn_infrastructure import rv_generic
from numpy import asarray, ndarray, float64, random, zeros

from typing import Literal, Callable


def abstract_shock_array(
    T: int,
    seed: int | None,
    dist: rv_generic,
    *dist_args: object,
    **dist_kwargs: object,
) -> ndarray:
    """
    Generate an array of shocks based on a specified distribution.

    Parameters:
    T (int): The number of time periods.
    dist: A scipy.stats distribution object (e.g., norm, t, uniform).
    *dist_args: Positional arguments for the distribution.
    **dist_kwargs: Keyword arguments for the distribution.

    Returns:
    np.ndarray: An array of shocks of length T.
    """
    state = random.RandomState(seed)
    shocks = dist.rvs(size=T, random_state=state, *dist_args, **dist_kwargs)  # type: ignore
    return asarray(shocks, dtype=float64)


def normal_shock_array(
    T: int, seed: int, mu: float = 0.0, sigma: float = 1.0
) -> ndarray:
    """
    Generate an array of normally distributed shocks.

    Parameters:
    T (int): The number of time periods.
    seed (int): Seed for the random number generator.
    mu (float): Mean of the normal distribution.
    sigma (float): Standard deviation of the normal distribution.

    Returns:
    np.ndarray: An array of normally distributed shocks of length T.
    """
    return abstract_shock_array(T, seed, norm, loc=mu, scale=sigma)


def t_shock_array(
    T: int, seed: int | None, df: float, loc: float = 0.0, scale: float = 1.0
) -> ndarray:
    """
    Generate an array of t-distributed shocks.

    Parameters:
    T (int): The number of time periods.
    seed (int): Seed for the random number generator.
    df (float): Degrees of freedom for the t-distribution.
    loc (float): Location parameter of the t-distribution.
    scale (float): Scale parameter of the t-distribution.

    Returns:
    np.ndarray: An array of t-distributed shocks of length T.
    """
    return abstract_shock_array(T, seed, t, df=df, loc=loc, scale=scale)


def uniform_shock_array(
    T: int, seed: int | None, loc: float = 0.0, scale: float = 1.0
) -> ndarray:
    """
    Generate an array of uniformly distributed shocks.

    Parameters:
    T (int): The number of time periods.
    seed (int): Seed for the random number generator.
    low (float): Lower bound of the uniform distribution.
    high (float): Upper bound of the uniform distribution.

    Returns:
    np.ndarray: An array of uniformly distributed shocks of length T.
    """
    return abstract_shock_array(T, seed, uniform, loc=loc, scale=scale)


def shock_placement(
    T: int, shock_spec: dict[int, float], shock_arr: ndarray = None
) -> ndarray:
    """
    Place shocks in a time series array based on a shock specification.

    Parameters:
    T (int): The number of time periods.
    shock_spec (dict): A dictionary where keys are time indices (0-based) and
                       values are shock scales (shock = scale * var_sigma at simulation time).

    Returns:
    np.ndarray: An array of shocks of length T with specified shocks placed.
    """

    if shock_arr is not None:
        shocks = shock_arr
    else:
        rdim = T
        shocks = zeros((rdim,), dtype=float64)

    for i, shock in shock_spec.items():
        shocks[i] = shock

    return shocks


class Shock:
    def __init__(
        self,
        T: int,
        dist: Literal["norm", "t", "uni"] | rv_generic | None = None,
        seed: int | None = 0,
        dist_args: tuple = (),
        dist_kwargs: dict = {},
        shock_arr: ndarray | None = None,
    ) -> None:

        self.T = T
        self.dist = dist
        self.seed = seed
        self.dist_args = dist_args
        self.dist_kwargs = dist_kwargs
        self.shock_arr = shock_arr

    # TODO: Pass through array if provided else generate based on dist

    def shock_generator(self) -> Callable[[float], ndarray]:
        assert self.dist is not None, "Distribution must be specified."
        assert (
            self.shock_arr is None
        ), "shock_arr is already provided. Please use place_shocks to mutate it."
        assert "scale" not in self.dist_kwargs, (
            "The generator function returns a callable that takes scale as an argument."
            " Please adjust `sig_` variables in the config to change the distribution scale."
            " Alternatively, the scale parameter in simulation and irf functions are multiplied directly with the shocks generated."
        )
        kwargs = self.dist_kwargs.copy()
        fun = lambda sigma: abstract_shock_array(
            self.T,
            self.seed,
            self._get_dist(),
            *self.dist_args,
            **{**kwargs, "scale": sigma},
        )

        return fun

    def place_shocks(self, shock_spec: dict[int, float]) -> ndarray:
        if self.shock_arr is not None:
            assert self.shock_arr.shape[0] == self.T, "shock_arr length must match T."

        return shock_placement(self.T, shock_spec, self.shock_arr)

    def _get_dist(self) -> rv_generic:
        dist = self.dist

        if dist == "norm":
            return norm
        elif dist == "t":
            return t
        elif dist == "uni":
            return uniform
        else:
            assert isinstance(
                dist, rv_generic
            ), "dist must be a valid scipy.stats distribution or a string identifier."
            return dist
