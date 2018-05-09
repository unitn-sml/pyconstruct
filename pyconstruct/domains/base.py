
import numpy as np

from ..utils import broadcast, hashkey, subdict

from abc import ABC, abstractmethod


__all__ = ['BaseDomain', 'InferenceError']


class BaseDomain(ABC):
    """The objects domain.

    The domain is the description of the objects and their features, and
    provides oracles for computing inference problems.

    This is the basic class

    Parameters
    ----------
    cache : dict-like
        A dict-like cache like those from the cachetools library.
    n_jobs : int
        The number of inference problems to solve in parallel.
    """
    def __init__(self, cache=None, n_jobs=1):
        self.cache = cache
        self.n_jobs = n_jobs

    def n_features(self, **kwargs):
        """Return the number of features in the feature vector.

        Parameters
        ----------
        kwargs
            Additional arguments needed by the domain file.
        """
        raise NotImplementedError

    @abstractmethod
    def _infer(self, *args, **kwargs):
        """Internal inference oracle.

        Subclasses should implement here the procedure to solve a sigle
        inference problem.
        """

    def phi(self, X, Y, **kwargs):
        """The feature vector of input X and output Y.

        Parameters
        ----------
        X : numpy.ndarray
            An array of input objects. The first dimension of the array should
            be the number of samples.
        Y : numpy.ndarray
            An array of output objects. The first dimension of the array should
            be the number of samples.
        kwargs
            Additional parameters.

        Returns
        -------
        phi : numpy.ndarray
            The array of feature vectors. The shape is (n_samples, n_features).
        """
        return self.infer(X, Y, problem='phi', **kwargs)

    def infer(self, *args, **kwargs):
        """Inference oracle.

        This is a generic inference method. It takes care of caching and
        parallelization. Subclasses should implement the _infer method, which is
        called for each sample passed to this method.

        Parameters
        ----------
        args : [numpy.ndarray]
            Input vectors of type numpy.ndarray. The first dimension of the
            vectors must be the number of samples (n_samples) and must be the
            same for all vectors.
        kwargs
            Additional parameters.

        Returns
        -------
        preds : numpy.ndarray
            The array of predictions. The array contains n_samples solutions.
        phis : numpy.ndarray
            The array of feature vectors corresponding to the predicted objects.
        """
        n_samples = args[0].shape[0]

        # Fetch cached predictions
        preds = [None] * n_samples
        keys = [None] * n_samples
        if self.cache is not None:
            for i, x in enumerate(zip(*args)):
                preds_keys[i] = hashkey(*x, **kwargs)
                if pred_keys[i] in self.cache:
                    preds[i] = self.cache[pred_keys[i]]

        # Compute remaining
        _idx = [idx for idx, pred in enumerate(preds) if pred is None]
        _args = [x[_idx] for x in args]
        _preds = broadcast(self._infer, *_args, n_jobs=self.n_jobs, **kwargs)

        # Save missing
        if self.cache is not None:
            for idx, pred in zip(_idx, _preds):
                self.cache[keys[idx]] = pred

        # Merge
        j = 0
        for i in range(n_samples):
            if preds[i] is None:
                preds[i] = _preds[j]
                j += 1

        return np.array(preds)


class InferenceError(RuntimeError):
    pass

