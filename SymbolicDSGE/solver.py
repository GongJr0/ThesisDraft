import sympy as sp  # type: ignore
from sympy import Symbol, Function, Eq, Expr
from sympy.core.relational import Relational  # type: ignore

import numpy as np
from numpy import float64, asarray, ndarray

from dataclasses import dataclass, asdict
from typing import Callable, Any, cast

from .model_config import ModelConfig


@dataclass(frozen=True)
class CompiledModel:
    var_names: list[str]
    idx: dict[str, int]
    objective_eqs: list[Expr]
    objective_funcs: list[Callable]

    equations: Callable
    n_state: int
    n_exog: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SolvedModel:
    compiled: CompiledModel
    policy: Any
    A: ndarray
    B: ndarray

    def sim(
        self, T: int, shocks: ndarray = None, x0: ndarray = None
    ) -> dict[str, ndarray]:
        raise NotImplementedError

    def irf(self, shock: str, T: int, scale: float = 1.0) -> dict[str, ndarray]:
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DSGESolver:
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.t = sp.Symbol("t", integer=True)

    def compile(
        self,
        *,
        variable_order: list[Function] = None,
        n_state: int = None,
        n_exog: int = None,
        params_order: list[str] = None,
    ) -> CompiledModel:

        conf = self.model_config
        t = self.t

        # Convert model to minimization problem
        obj = [sp.simplify(eq.lhs - eq.rhs) for eq in conf.equations.model]

        shifted = [self._offest_lags(o, t) for o in obj]

        # Deterministic var order
        if not variable_order:
            var_order: list[str] = [
                v.func.__name__ if hasattr(v, "func") else v.__name__
                for v in conf.variables
            ]
            var_order = [v.__name__ for v in conf.variables]
        else:
            var_order = [
                v.__name__ if hasattr(v, "func") else v for v in variable_order
            ]

        name_to_func = {v.__name__: v for v in conf.variables}
        missing = [v for v in var_order if v not in name_to_func]
        if missing:
            raise ValueError(
                f"The following variables in var_order do not exist in the model: {missing}"
            )

        var_funcs = [name_to_func[name] for name in var_order]
        idx = {name: i for i, name in enumerate(var_order)}

        for i, obj in enumerate(shifted):
            bad = self._bad_time_offsets(obj, var_funcs, t)
            if bad:
                raise ValueError(
                    f"Equation {i} has bad time offsets {bad}. Only offsets of 0 and 1 are allowed."
                )

        # Substitutions
        cur_syms = [Symbol(f"cur_{n}") for n in var_order]
        fwd_syms = [Symbol(f"fwd_{n}") for n in var_order]

        subs_map = {}
        for name, f, cur, fwd in zip(var_order, var_funcs, cur_syms, fwd_syms):

            subs_map[f(t)] = cur
            subs_map[f(t + 1)] = fwd

        if not params_order:
            params_order = [p.name for p in conf.parameters]

        name_to_param = {p.name: p for p in conf.parameters}
        p_missing = [p for p in params_order if p not in name_to_param]
        if p_missing:
            raise ValueError(f"params_order contains unknown parameters: {p_missing}")
        params = [name_to_param[name] for name in params_order]

        compiled: list[Expr] = [sp.simplify(o.subs(subs_map)) for o in shifted]
        lambda_args = [*cur_syms, *fwd_syms, *params]
        funcs = [sp.lambdify(lambda_args, c, modules="numpy") for c in compiled]

        def equations(
            fwd: ndarray, cur: ndarray, par: dict[str, float] | ndarray
        ) -> ndarray:
            fwd = np.asarray(fwd, dtype=float)
            cur = np.asarray(cur, dtype=float)

            if isinstance(par, dict):
                par_vec = np.array([par[p.name] for p in params], dtype=float)
            else:
                par_vec = np.asarray(par, dtype=float)
                if par_vec.shape[0] != len(params):
                    raise ValueError(
                        f"Parameter vector length {par_vec.shape[0]} != {len(params)}"
                    )

            out = np.empty(len(funcs), dtype=float)
            # scalar call per equation; tiny systems so this is fine
            for i, f in enumerate(funcs):
                out[i] = float(f(*fwd, *cur, *par_vec))
            return out

        if n_state is None or n_exog is None:
            raise ValueError(
                "For linearsolve backend you must provide n_state and n_exog explicitly for now. "
                "Later we can infer them, but explicit is safer for the first iteration."
            )

        return CompiledModel(
            var_names=var_order,
            idx=idx,
            objective_eqs=compiled,
            objective_funcs=funcs,
            equations=equations,
            n_state=int(n_state),
            n_exog=int(n_exog),
        )

    @staticmethod
    def _min_time_offset(expr: Expr, t: Symbol) -> int:
        offs = []
        for call in expr.atoms(Function):
            if not call.args:
                continue

            arg0 = call.args[0]
            if arg0.free_symbols and t in arg0.free_symbols:
                k = sp.simplify(arg0 - t)
                if k.is_Integer:
                    offs.append(int(k))
        return min(offs) if offs else 0

    def _offest_lags(self, obj: Expr, t: Symbol) -> Expr:
        min_off = self._min_time_offset(obj, t)

        if min_off < 0:
            return sp.simplify(obj.subs(t, t - min_off))
        return obj

    @staticmethod
    def _bad_time_offsets(expr: Expr, var_funcs: list[Function], t: Symbol) -> set[int]:
        allowed = {0, 1}
        bad: set[int] = set()

        for call in expr.atoms(sp.Function):
            if (
                call.func not in [vf.func for vf in var_funcs]
                and call.func not in var_funcs
            ):
                pass

            if not call.args:
                continue

            arg0 = call.args[0]
            if arg0.free_symbols and t in arg0.free_symbols:
                k = sp.simplify(arg0 - t)
                if k.is_integer:
                    kk = int(k)
                    if kk not in allowed:
                        bad.add(kk)
        return bad
