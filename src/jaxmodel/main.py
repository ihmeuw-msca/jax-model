from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Optional

import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import grad, jacfwd, jacrev, jit
from numpy.typing import NDArray
from scipy.optimize import Bounds, minimize


@dataclass
class Formula:

    obs_mean: str = "obs_mean"
    obs_se: str = "obs_se"
    covs: list[str] = field(default_factory=list)
    link: Callable[[NDArray], NDArray] = jit(lambda x: x)

    def __post_init__(self) -> None:
        if not self.covs:
            self.covs = ["intercept"]


class Model:

    def __init__(self, formula: Formula, data: Optional[pd.DataFrame] = None) -> None:
        self.formula = formula
        self.attach_data(data)

        self.gradient = jit(grad(self.objective))
        self.hessian = jit(jacfwd(jacrev(self.objective)))

        self.opt_result = None

    def attach_data(self, data: Optional[pd.DataFrame]) -> None:
        if data is not None:
            data = data.copy()
            if "intercept" not in data:
                data["intercept"] = 1.0
            self._data = (
                data[self.formula.obs_mean].to_numpy(),
                data[self.formula.obs_se].to_numpy(),
                data[self.formula.covs].to_numpy(),
            )

    def detach_data(self) -> None:
        self._data = None

    @property
    def data(self) -> Optional[tuple]:
        return getattr(self, "_data", None)

    @partial(jit, static_argnums=(0,))
    def objective(self, x: NDArray) -> float:
        obs_mean, obs_se, covs = self.data
        prediction = self.formula.link(jnp.dot(covs, x[:-1]))
        residual = obs_mean - prediction
        variance = x[-1] + obs_se**2
        return 0.5*jnp.sum(residual**2/variance + jnp.log(variance))

    def fit(self, x0: Optional[NDArray] = None, options: Optional[dict] = None) -> None:
        if self.data is None:
            raise ValueError("please attach data before fit the model")

        num_covs = len(self.formula.covs)
        if x0 is None:
            x0 = np.zeros(num_covs + 1)
        bounds = Bounds(
            lb=[-np.inf]*num_covs + [0.0],
            ub=[np.inf]*(num_covs + 1),
        )

        self.opt_result = minimize(
            self.objective,
            x0,
            method="trust-constr",
            jac=self.gradient,
            hess=self.hessian,
            bounds=bounds,
            options=options
        )
