
import numpy as np

from ..utils import broadcast, hashkey, subdict

from abc import ABC, abstractmethod


__all__ = ['BaseDomain', 'InferenceError']


class BaseDomain(ABC):
    """The structured objects domain.

    The domain contain the description of the objects and provides oracles for
    computing inference problems.

    This is the basic class from which domains should inherit. In most cases,
    when working with MiniZinc domains, it should not be necessary to create a
    custom domain, but it is always possible to create a new Domain class for
    e.g. using an ad-hoc inference algorithm to make predictions more efficient.

    In the future we might introduce additional domains and inference methods.

    Parameters
    ----------
    cache : dict-like
        A dict-like cache like those from the `cachetools` library.
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
            Additional parameters to pass to each call of _infer.

        Returns
        -------
        preds : numpy.ndarray
            The array of predictions. The array contains n_samples solutions.
        """
        n_samples = args[0].shape[0]

        # Fetch cached predictions
        preds = [None] * n_samples
        keys = [None] * n_samples
        if self.cache is not None:
            for i, x in enumerate(zip(*args)):
                keys[i] = hashkey(*x, **kwargs)
                if keys[i] in self.cache:
                    preds[i] = self.cache[keys[i]]

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
    """ Raised when an error with an inference procedure occurs """
    pass

