
import warnings
import numpy as np

from sklearn.externals.joblib import Parallel, delayed


__all__ = [
    'broadcast', 'asarrays', 'array2str', 'batches'
]


def broadcast(f, *args, n_jobs=1, **kwargs):
    """Applies f to each element of the input vectors."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.array(Parallel(n_jobs)(
            delayed(f)(*x, **kwargs) for x in zip(*args)
        ))


def isarray(x):
    """Check whether an object is a non-empty array."""
    return isinstance(x, np.ndarray) and len(x.shape) > 0


def asarrays(*args):
    """Returns the input objects as one-element arrays"""
    return tuple([np.array([x]) if not isarray(x) else x for x in args])


def array2str(array):
    """Pretty formatter of a numpy array for one-line logging."""
    return np.array2string(
        array, max_line_width=np.inf, separator=',', precision=None,
        suppress_small=None
    ).replace('\n', '')


def _batch_sizes(data_size, n_batches):
    batch_sizes = []
    for _ in range(n_batches):
        batch_sizes.append(
            data_size // n_batches + (data_size % n_batches) // n_batches
        )
        data_size -= batch_sizes[-1]
        n_batches -= 1
    return batch_sizes


def batches(*args, n_batches=None, batch_size=None):
    """Returns a generator of batches.

    Takes an arbitrary number of array-like objects and splits them into
    batches. One can either require a certain number of batches or a certain
    batch size.

    Parameters
    ----------
    *args
        A list of numpy.ndarrays as input parameters.
    n_batches : int
        The number of batches to split the data into.
    batch_size : int
        The size of the batches to split the data into.

    Returns
    -------
    generator
        A generator of batches.
    """
    data_size = len(args[0])
    if n_batches is not None:
        batch_sizes = _batch_sizes(data_size, n_batches)
    elif batch_size is not None:
        n_batches = data_size // batch_size + (data_size % batch_size > 0)
        batch_sizes = [batch_size] * n_batches
    else:
        raise ValueError('Need to provide either n_batches or batch_size')

    batch_indices = [0] + np.cumsum(batch_sizes).tolist()
    for start, end in zip(batch_indices[:-1], batch_indices[1:]):
        yield [np.array(arg[start:end]) for arg in args]

