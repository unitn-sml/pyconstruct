
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

    The weight vector :math:`\boldsymbol{w}` is mandatory.

    The parameter `features` is a vector containing part of the features of the
    model that are directly handled into Python code. They could be, for
    instance, the result of some feature learning procedure. The rest of the
    feature vector is specified directly into the domain. The two arrays are
    complementary and, when using the `solve` macro from `pyconstruct.pmzn`,
    they are concatenated and used for computing the complete feature vector
    :math:`\boldsymbol{phi}`.  The size of the weights vector `w` must be equal
    to the sum of the size of the two feature arrays. Using this parameter is
    optional and in most cases it suffices to encode features into the pmzn
    domain.

    Parameters
    ----------
    domain : Domain
        The underlying domain.
    w : np.ndarray
        The weight vector.
    features : np.ndarray
        An optional array of additional features.
    """
    def __init__(self, domain=None, w=None, features=None, **kwargs):
        super().__init__(domain=domain, **kwargs)
        self.w = w
        self.features = features

    def _validate_params(self):
        super()._validate_params()
        if not isinstance(self.w, np.ndarray):
            raise ValueError('w must be a numpy array')
        if len(self.w.shape) != 1:
            raise ValueError('w must be a one-dimentional array')
        if self.feature is not None:
            if not isinstance(self.features, np.ndarray):
                raise ValueError('features must be a numpy array')
            if len(self.features.shape) != 1:
                raise ValueError('features must be a one-dimentional array')

    @property
    def parameters(self):
        return {**super().parameters, 'w': self.w, 'features': self.features}

    def decision_function(self, X, Y):
        self._validate_params()
        return np.inner(self.w, self.phi(X, Y))

