
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
    return HashableArgs(*args, **kwargs)


def typedkey(*args, **kwargs):
    return TypedHashableArgs(*args, **args)

