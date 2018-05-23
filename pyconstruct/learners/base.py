
from ..models import BaseModel
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator


__all__ = ['BaseLearner']


class BaseLearner(BaseEstimator, ABC):
    """A basic learning model class.

    A learner fits a model with some data over some given domain.  If only the
    domain is given, a default model is used. If only the model is given, it
    must also contain a domain to make predictions with. If both are given, the
    domain of the model (if it has one) will be overwritten by the domain given
    to the learner.

    Arguments
    ---------
    domain : BaseDomain
        The domain of the data.
    model : BaseModel
        The model the learner should fit.
    """
    def __init__(self, domain=None, model=None, **kwargs):
        self.domain = domain
        self.model = model

    def _model(self, default=None):
        if self.model is None:
            if self.domain is None:
                raise ValueError('Either domain or model must be given')
            model_class = default if default is not None else BaseModel
            self.model = model_class(self.domain)
        elif self.domain is not None:
            self.model.domain = self.domain
        return self.model

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
        return self._model.phi(X, Y, **kwargs)

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
        return self._model.predict(X, *args, **kwargs)

    def decision_function(self, X, Y, **kwargs):
        return self._model.decision_function(X, Y, **kwargs)

    def loss(self, X, Y, Y_pred, **kwargs):
        return self._model.loss(X, Y, Y_pred, **kwargs)

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

