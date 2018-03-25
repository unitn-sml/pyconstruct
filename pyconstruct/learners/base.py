
import numpy as np

from ..models import BaseModel
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator


__all__ = ['BaseLearner']


class BaseLearner(BaseEstimator, ABC):
    """A basic learning model class.

    A model performs learning and inference in a given domain. Subclasses should
    implement more specific types of learning useful for particular problems.

    Arguments
    ---------
    domain : BaseDomain
        The domain.
    """
    def __init__(self, *, domain=None):
        if domain is None:
            raise ValueError('Need to specify a domain.')
        self.domain = domain

    @property
    def model(self):
        """The predictive model the algorithm has learned."""
        if not hasattr(self, 'model_'):
            return BaseModel(self.domain)
        return self.model_

    def predict(self, X, **kwargs):
        """Computes the prediction of the current model for the given input.

        Parameters
        ----------
        X : numpy.ndarray
            An array of input examples. The first dimension must be the number
            of samples.

        Returns
        -------
        numpy.ndarray
            The array of predicted objects.
        """
        return self.model.predict(X, **kwargs)

    def score(self, X, Y_true, Y_pred=None, **kwargs):
        """Compute the score as the average loss over the examples.

        This method is needed for scikit-learn estimation in GridSearchCV and
        other model selection methods.

        Parameters
        ----------
        X : numpy.ndarray
            An array of input examples. The first dimension must be the number
            of samples.
        Y_true : numpy.ndarray
            An array of true output objects.
        Y_pred : numpy.ndarray
            An array of predicted object.

        Returns
        -------
        score : float
            The score of the model over the examples.
        """
        if Y_pred is None:
            Y_pred = self.predict(X, **kwargs)
        return (- np.array(self.loss(X, Y_true, Y_pred))).mean()

    def decision_function(self, X, Y):
        return self.model.decision_function(X, Y)

    def loss(self, X, Y_true, Y_pred):
        return self.model.loss(X, Y_true, Y_pred)

    @abstractmethod
    def partial_fit(self, X, Y, Y_pred=None):
        """Updates the current model with data (X, Y).

        Parameters
        ----------
        X : numpy.ndarray
            Input examples. The first dimension must be the batch size.
        Y : numpy.ndarray
            Output objects. The first dimension must be the batch size. This
            must coincide with batch size for X.
        Y_pred : numpy.ndarray
            Predictions of the algorithm. The first dimension must be the batch
            size. This must coincide with batch size for X. If None, either not
            needed or done internally.

        Returns
        -------
        self
        """

    @abstractmethod
    def fit(self, X, Y, Y_pred=None):
        """Fit a model with data (X, Y).

        Parameters
        ----------
        X : numpy.ndarray
            Input examples. The first dimension must be the dataset size.
        Y : numpy.ndarray
            Output objects. The first dimension must be the dataset size. This
            must coincide with batch size for X.
        Y_pred : numpy.ndarray
            Predictions of the algorithm. The first dimension must be the batch
            size. This must coincide with dataset size for X. If None, either
            not needed or done internally.

        Returns
        -------
        self
        """

