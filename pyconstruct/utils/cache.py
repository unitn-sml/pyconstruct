
from sklearn.externals import joblib

try:
    from cachetools.keys import _kwmark
except ImportError:
    _kwmark = (object(),)


__all__ = ['hashkey', 'typedkey']


class HashableArgs:

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self._hashkey = self.args + sum(sorted(self.kwargs.items()), _kwmark)
        self._hashvalue = None

    def __eq__(self, other):
       if len(self._hashkey) != len(other._hashkey):
           return False

       for arg1, arg2 in zip(self._hashkey, other._hashkey):
           if arg1 != arg2:
               return False
       return True

    # def __lt__(self, other):
    #    return hash(self) < hash(other)

    # def __gt__(self, other):
    #    return hash(self) > hash(other)

    def __hash__(self):
        if self._hashvalue is None:
            self._hashvalue = int(joblib.hash(self._hashkey), 16)
        return self._hashvalue


class TypedHashableArgs:

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._hashkey += tuple(type(v) for v in args)
        self._hashkey += tuple(type(v) for _, v in sorted(kwargs.items()))


def hashkey(*args, **kwargs):
    """Return an hasable object out of the input parameters."""
    return HashableArgs(*args, **kwargs)


def typedkey(*args, **kwargs):
    """Return an hasable object out of the input parameters.

    The type of the objects is taken into account.
    """
    return TypedHashableArgs(*args, **args)

