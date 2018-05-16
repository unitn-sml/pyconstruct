
import numpy as np

from ..domains import BaseDomain
from ..utils import broadcast


__all__ = ['BaseModel', 'LinearModel']


class BaseModel:
    """Base model.

    Parameters
    ----------
    domain : Domain
        The underlying domain.
    """
    def __init__(self, domain=None, **kwargs):
        self.domain = domain

    def _validate_params(self):
        if not isinstance(self.domain, BaseDomain):
            raise ValueError('domain must be an instance of BaseDomain')

    @property
    def parameters(self):
        """A dictionary of parameters of the model."""
        self._validate_params()
        return {'type': type(self).__name__}

    def phi(self, X, Y):
        self._validate_params()
        return self.domain.phi(X, Y, model=self)

    def predict(self, X, *args, **kwargs):
        """Makes a prediction based on the current model."""
        self._validate_params()
        return self.domain.infer(X, *args, model=self, **kwargs)

    def decision_function(self, X, Y):
        """Computes the score assigned to the (x, y) by the model."""
        self._validate_params()
        return 0.0

    def margin(self, X, Y, Y_pred):
        """Computes the margin of the current model for the given data."""
        Y_f = self.decision_function(X, Y)
        Y_pred_f = self.decision_function(X, Y_pred)
        return Y_f - Y_pred_f

    def loss(self, X, Y, Y_pred):
        """Computes the loss of the predictions with respect the model."""
        return - self.margin(X, Y, Y_pred)


class LinearModel(BaseModel):
    r"""A linear model.

    Represents a linear model of the type:
    :math:`F(x, y) = \langle \boldsymbol{w}, \boldsymbol{\phi}(x, y) \rangle`.

    The only parameter it needs is the weight vector :math:`\boldsymbol{w}`.

    Parameters
    ----------
    domain : Domain
        The underlying domain.
    w : np.ndarray
        The weight vector.
    """
    def __init__(self, domain=None, w=None, **kwargs):
        super().__init__(domain=domain, **kwargs)
        self.w = w

    def _validate_params(self):
        super()._validate_params()
        if not isinstance(self.w, np.ndarray):
            raise ValueError('w must be an numpy array')
        if len(self.w.shape) > 1:
            raise ValueError('w must be a one-dimentional array')

    @property
    def parameters(self):
        return {**super().parameters, 'w': self.w}

    def decision_function(self, X, Y):
        self._validate_params()
        return np.inner(self.w, self.phi(X, Y))

