from dataclasses import dataclass
from functools import partial
from typing import Optional

import jax.numpy as jnp
import numpy as np
import pandas as pd
from anml.data.component import Component
from jax import grad, hessian, jit
from numpy.typing import NDArray
from scipy.optimize import minimize

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
            self.formula.obs_mean.attach(data)
            self.formula.obs_se.attach(data)
            self.formula.mu.attach(data)
            self.formula.alpha.attach(data)

    def clear(self) -> None:
        self.formula.obs_mean.clear()
        self.formula.obs_se.clear()
        for v in self.formula.mu.variables:
            v.component.clear()
        self.formula.mu.design_mat = None
        for v in self.formula.alpha.variables:
            v.component.clear()
        self.formula.alpha.design_mat = None

    @property
    def data(self) -> Optional[tuple]:
        return getattr(self, "_data", None)

    @partial(jit, static_argnums=(0,))
    def objective(self, x: NDArray) -> float:
        obs_mean = self.formula.obs_mean.value
        obs_se = self.formula.obs_se.value

        mu = self.formula.mu.get_params(x[:self.formula.mu.size])
        alpha = self.formula.alpha.get_params(x[self.formula.mu.size:])

        residual = obs_mean - mu
        variance = alpha + obs_se**2
        return 0.5*jnp.sum(residual**2/variance + jnp.log(variance))

    def fit(self, x0: Optional[NDArray] = None, options: Optional[dict] = None) -> None:
        size = self.formula.mu.size + self.formula.alpha.size
        if x0 is None:
            x0 = np.zeros(size)

        self.opt_result = minimize(
            self.objective,
            x0,
            method="trust-constr",
            jac=self.gradient,
            hess=self.hessian,
            options=options
        )
