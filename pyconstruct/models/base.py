
import numpy as np

from ..utils import broadcast


__all__ = ['BaseModel', 'LinearModel']


class BaseModel:
    """Base model.

    Parameters
    ----------
    domain : Domain
        The underlying domain.
    """
    def __init__(self, domain):
        self.domain = domain

    @property
    def parameters(self):
        """A dictionary of parameters of the model."""
        return {'type': type(self).__name__}

    def phi(self, X, Y):
        return self.domain.phi(X, Y, model=self)

    def predict(self, X, **kwargs):
        """Makes a prediction based on the current model."""
        return self.domain.infer(X, model=self, **kwargs)

    def decision_function(self, X, Y):
        """Computes the score assigned to the (x, y) by the model."""
        return 0.0

    def margin(self, X, Y_true, Y_pred):
        """Computes the margin of the current model for the given data."""
        f_true = self.decision_function(X, Y_true)
        f_pred = self.decision_function(X, Y_pred)
        return f_true - f_pred

    def loss(self, X, Y_true, Y_pred):
        """Computes the loss of the predictions with respect the model."""
        return - self.margin(X, Y_true, Y_pred)


class LinearModel(BaseModel):
    """A linear model.

    Represents a linear model of the type:
    .. math:: \langle \boldsymbol{w}, \boldsymbol{\phi}(x, y) \rangle

    The only parameter it needs is the weight vector :math:`\boldsymbol{w}`.

    Parameters
    ----------
    domain : Domain
        The underlying domain.
    w : np.ndarray
        The weight vector.
    """
    def __init__(self, domain, w):
        super().__init__(domain)
        self.w = w

    @property
    def parameters(self):
        return {'w': self.w, **super().parameters}

    def decision_function(self, X, Y):
        return np.inner(self.w, self.phi(X, Y))

