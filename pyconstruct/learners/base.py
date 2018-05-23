
import numpy as np

from ..models import BaseModel
from ..domains import BaseDomain
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
    def __init__(self, domain=None, model=None, **kwargs):
        self.domain = domain
        self.model = model

    def phi(self, X, Y, **kwargs):
        """Computes the feature vector for the given input and output objects.

        Parameters
        ----------
        X : numpy.ndarray
            An array of input examples. The first dimension must be the number
            of samples.
        Y : numpy.ndarray
            An array of output objects.

        Returns
        -------
        numpy.ndarray
            The array of feature vectors.
        """
        return self.model.phi(X, Y, **kwargs)

    def predict(self, X, *args, **kwargs):
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
        return self.model.predict(X, *args, **kwargs)

    def decision_function(self, X, Y):
        return self.model.decision_function(X, Y)

    def loss(self, X, Y, Y_pred):
        return self.model.loss(X, Y, Y_pred)

    def score(self, X, Y, Y_pred=None, **kwargs):
        """Compute the score as the average loss over the examples.

        This method is needed for scikit-learn estimation in GridSearchCV and
        other model selection methods.

        Parameters
        ----------
        X : numpy.ndarray
            An array of input examples. The first dimension must be the number
            of samples.
        Y : numpy.ndarray
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
        return (- self.loss(X, Y, Y_pred)).mean()

    @abstractmethod
    def partial_fit(self, X, Y, Y_pred=None, **kwargs):
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
    def fit(self, X, Y, Y_pred=None, **kwargs):
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

