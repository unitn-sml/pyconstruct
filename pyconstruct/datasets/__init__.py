
import os
import pickle

from . import utils
from sklearn.utils import Bunch


# List of available datasets
DATASETS = list(utils.SOURCES.keys())


def load(dataset, *, base=None, fetch=True, force=False):
    """Load a dataset.

    This method loads one of the predefined dataset. The list of available
    datasets can be found in the `DATASETS` variable.

    The returned dataset is preprocessed in order to be usable out-of-the-box by
    the Weaver algorithms. The preprocessed version is automatically cached.

    Parameters
    ----------
    dataset : str
        The name of the dataset.
    base : str
        The base directory where to look for the dataset or to fetch it into.
        Default is a system-dependent data directory.
    fetch : bool
        Whether to fetch the dataset in case it is not found.
    force : bool
        Whether to force the preprocessing of the dataset.

    Returns
    -------
    dataset : sklearn.utils.Bunch
        A collection of properties of the dataset.
    """
    if dataset not in utils.SOURCES:
        raise ValueError('Invalid dataset.')

    cache_dir = utils.data_dir(dataset, base)
    cache_file = os.path.join(cache_dir, '{}.pickle'.format(dataset))
    if os.path.exists(cache_file) and not force:
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    paths = utils.get_paths(dataset, base)
    if not utils.exist(paths):
        if fetch:
            paths = utils.fetch(dataset, base)
        else:
            raise RuntimeError('Dataset not found, need to fetch it first.')

    module = __import__('.'.join([__name__, dataset]), fromlist=['load_data'])
    X, Y, *args = module.load_data(paths)

    if len(args) > 0:
        kwargs = args[0]

    descr = None if not hasattr(module, 'DESCR') else module.DESCR
    dataset = Bunch(data=X, target=Y, DESCR=descr, **kwargs)
    with open(cache_file, 'wb') as f:
        pickle.dump(dataset, f)
    return dataset


def load_ocr(**kwargs):
    """Convenience function for loading the OCR dataset."""
    return load('ocr', **kwargs)


def load_conll00(**kwargs):
    """Convenience function for loading the CoNLL00 dataset."""
    return load('conll00', **kwargs)

