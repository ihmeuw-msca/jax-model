from dataclasses import dataclass
from functools import partial
from typing import Optional

import jax.numpy as jnp
import numpy as np
import pandas as pd
from anml.data.component import Component
from jax import grad, hessian, jit
from numpy.typing import NDArray
from scipy.linalg import block_diag
from scipy.optimize import Bounds, LinearConstraint, minimize

from jaxmodel.parameter import JaxParameter


@dataclass
class Formula:

    obs_mean: Component
    obs_se: Component
    mu: JaxParameter
    alpha: JaxParameter


class Model:

    def __init__(self, formula: Formula, data: Optional[pd.DataFrame] = None) -> None:
        self.formula = formula
        self.attach(data)

        self.gradient = jit(grad(self.objective))
        self.hessian = jit(hessian(self.objective))

        self.opt_result = None

    def attach(self, data: Optional[pd.DataFrame]) -> None:
        if data is not None:
            for attr in vars(self.formula).values():
                attr.attach(data)

    def clear(self) -> None:
        for attr in vars(self.formula).values():
            attr.clear()

    @property
    def data(self) -> Optional[tuple]:
        return getattr(self, "_data", None)

    @partial(jit, static_argnums=(0,))
    def objective(self, x: NDArray) -> float:
        obs_mean = self.formula.obs_mean.value
        obs_se = self.formula.obs_se.value

        x_mu = x[:self.formula.mu.size]
        x_alpha = x[self.formula.mu.size:]

        mu = self.formula.mu.get_params(x_mu)
        alpha = self.formula.alpha.get_params(x_alpha)

        residual = obs_mean - mu
        variance = alpha + obs_se**2

        value = 0.5*jnp.sum(residual**2/variance + jnp.log(variance))
        value += self.formula.mu.prior_objective(x_mu)
        value += self.formula.alpha.prior_objective(x_alpha)

        return value

    def get_bounds(self) -> Bounds:
        bounds = np.hstack([
            self.formula.mu.prior_dict["direct"]["UniformPrior"].params,
            self.formula.alpha.prior_dict["direct"]["UniformPrior"].params,
        ])
        return Bounds(lb=bounds[0], ub=bounds[1])

    def get_constraints(self) -> Optional[list[LinearConstraint]]:
        mat = block_diag(
            self.formula.mu.prior_dict["linear"]["UniformPrior"].mat,
            self.formula.alpha.prior_dict["linear"]["UniformPrior"].mat,
        )
        bounds = np.hstack([
            self.formula.mu.prior_dict["linear"]["UniformPrior"].params,
            self.formula.alpha.prior_dict["linear"]["UniformPrior"].params,
        ])
        if mat.size == 0 and bounds.size == 0:
            return None
        return [LinearConstraint(A=mat, lb=bounds[0], ub=bounds[1])]

    def fit(self, x0: Optional[NDArray] = None, options: Optional[dict] = None) -> None:
        size = self.formula.mu.size + self.formula.alpha.size
        if x0 is None:
            x0 = np.zeros(size)

        bounds = self.get_bounds()
        constraints = self.get_constraints()

        self.opt_result = minimize(
            self.objective,
            x0,
            method="trust-constr",
            jac=self.gradient,
            hess=self.hessian,
            bounds=bounds,
            constraints=constraints,
            options=options
        )

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        self.clear()
        self.formula.mu.attach(df)
        self.formula.alpha.attach(df)

        x_mu = self.opt_result.x[:self.formula.mu.size]
        x_alpha = self.opt_result.x[self.formula.mu.size:]
        df["mu"] = self.formula.mu.get_params(x_mu)
        df["alpha"] = self.formula.alpha.get_params(x_alpha)

        self.clear()

        return df
