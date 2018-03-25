
__all__ = ['dict2str', 'subdict', 'dictsplit']


def dict2str(d):
    """Pretty formatter of a dictionary for one-line logging."""
    return str(sorted(d.items()))


def subdict(d, keys=None, nokeys=None):
    """Returns a subdictionary.

    Parameters
    ----------
    d : dict
        A dictionary.
    keys : list or set
        The set of keys to include in the subdictionary. If None use all keys.
    nokeys : list or set
        The set of keys to exclude from the subdictionary. If None no key is
        excluded.
    """
    keys = set(keys if keys else d.keys())
    nokeys = set(nokeys if nokeys else [])
    return {k: v for k, v in d.items() if k in (keys - nokeys)}


def dictsplit(d, keys):
    """Split a dictionary in two given a set of keys.

    The first dictionary contains only the provided keys, whereas the second
    contains all the remaining keys.

    Parameters
    ----------
    d : dict
        A dictionary.
    keys : list or set
        The set of keys to split the dictionary on.
    """
    return subdict(d, keys=keys), subdict(d, nokeys=keys)

