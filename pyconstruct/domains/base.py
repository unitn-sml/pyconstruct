
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
        pred : numpy.ndarray
            The array of predictions. The array contains n_samples solutions.
        """
        n_samples = args[0].shape[0]

        pred_problem = 'problem' not in kwargs or kwargs['problem'] in [None, 'map', 'loss_augmented_map']

        # Check cache
        if self.cache is not None:
            keys = []
            preds = []
            if pred_problem:
                phi_keys = []
            for x in zip(*args):
                key = hashkey(*x, **kwargs)
                keys.append(key)
                if pred_problem:
                    phi_key = hashkey(*x, **{**subdict(kwargs, nokeys=['model']), 'problem': 'phi'})
                    phi_keys.append(phi_key)
                preds.append(None)
                if key in self.cache:
                    preds[-1] = self.cache[key]
        else:
            preds = [None] * n_samples

        _idx = [i for i in range(n_samples) if preds[i] is None]
        _args = [x[_idx] for x in args]
        _preds = broadcast(self._infer, *_args, n_jobs=self.n_jobs, **kwargs)

        if pred_problem:
            _preds, _phis = list(zip(*_preds))

        # Save missing
        if self.cache is not None:
            for i, pred in zip(_idx, _preds):
                self.cache[keys[i]] = pred
            if pred_problem:
                for i, phi in zip(_idx, _phis):
                    if phi is not None:
                        self.cache[phi_keys[i]] = phi

        # Merge
        j = 0
        for i in range(n_samples):
            if preds[i] is None:
                preds[i] = _preds[j]
                j += 1

        return np.array(preds)


class InferenceError(RuntimeError):
    pass

