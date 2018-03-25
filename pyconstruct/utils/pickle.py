
import pickle


__all__ = ['save', 'load']


def save(obj, path):
    """Save a Python object into a Pickle file at the given path."""
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load(path):
    """Loads a Python object from a Pickle file at the given path."""
    with open(path, 'rb') as f:
        return pickle.load(f)

