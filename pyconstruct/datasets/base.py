
import os
import pickle

from . import utils
from sklearn.utils import Bunch


__all__ = ['DATASETS', 'load', 'load_ocr', 'load_conll00', 'load_equations']


# List of available datasets
DATASETS = list(utils.SOURCES.keys())


def load(dataset, *, base=None, fetch=True, force=False, remove_raw=False):
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
    remove_raw : bool
        Wether to remove the download raw files.

    Returns
    -------
    dataset : sklearn.utils.Bunch
        A collection of properties of the dataset.
    """
    if dataset not in utils.SOURCES:
        raise ValueError('Invalid dataset.')

    cache_dir = utils.data_dir(dataset, base)
    cache_file = os.path.join(cache_dir, '{}_cached.pickle'.format(dataset))
    if os.path.exists(cache_file) and not force:
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    paths = utils.get_paths(dataset, base)
    if not utils.exist(paths):
        if fetch:
            utils.fetch(dataset, base, remove_raw=remove_raw)
        else:
            raise RuntimeError('Dataset not found, need to fetch it first.')

    module = __import__(
        '.'.join([__package__, dataset]), fromlist=['load_data']
    )
    X, Y, *args = module.load_data(paths)

    if remove_raw:
        for path in paths:
            os.remove(path)

    if len(args) > 0:
        kwargs = args[0]
    else:
        kwargs = {}

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

def load_equations(**kwargs):
    """Convenience function for loading the equations dataset."""
    return load('equations', **kwargs)

