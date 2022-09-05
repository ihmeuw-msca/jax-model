from typing import Callable, Optional

import jax.numpy as jnp
import pandas as pd
from anml.parameter.main import Parameter
from jax import jit
from numpy.typing import NDArray


class JaxParameter(Parameter):

    @Parameter.transform.setter
    def transform(self, transform: Optional[Callable]):
        if transform is None:
            transform = jit(lambda x: x)
        if not callable(transform):
            raise TypeError("JaxParameter transform must be callable.")
        self._transform = transform

    def get_params(self, x: NDArray, df: Optional[pd.DataFrame] = None) -> NDArray:
        if df is not None:
            self.attach(df)
        if self.design_mat is None:
            raise ValueError("Must provide a data frame to attach data.")
        y = jnp.dot(self.design_mat, x)
        if self.offset is not None:
            y += self.offset.value
        return self.transform(y)
