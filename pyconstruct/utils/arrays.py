
import numpy as np

from sklearn.externals.joblib import Parallel, delayed


__all__ = [
    'broadcast', 'asarrays', 'array2str', 'batches'
]


def broadcast(f, *args, n_jobs=1, **kwargs):
    """Applies f to each element of the input vectors."""
    return np.array(Parallel(n_jobs)(
        delayed(f)(*x, **kwargs) for x in zip(*args)
    ))


def asarrays(*args):
    return tuple([np.array([x]) for x in args])


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
    if not check_iterables(*args):
        args = [[arg] for arg in args]
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

