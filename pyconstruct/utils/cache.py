
from sklearn.externals import joblib
from collections.abc import Mapping

try:
    from cachetools.keys import _kwmark
except ImportError:
    _kwmark = (object(),)


__all__ = ['hashkey', 'typedkey']


class HashableArgs:

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        _hashkey = self.args + sum(sorted(self.kwargs.items()), _kwmark)
        self._hashkey = []
        for arg in _hashkey:
            if isinstance(arg, dict):
                self._hashkey += list(sum(sorted(arg.items()), _kwmark))
            else:
                self._hashkey.append(arg)
        self._hashkey = tuple(self._hashkey)
        self._hashvalue = None

    def __eq__(self, other):
       if len(self._hashkey) != len(other._hashkey):
           return False

       for arg1, arg2 in zip(self._hashkey, other._hashkey):
           if arg1 != arg2:
               return False
       return True

    def __lt__(self, other):
        for i in range(max(len(self._hashkey), len(other._hashkey))):
            o1, o2 = None, None
            if i < len(self._hashkey):
                o1 = self._hashkey[i]
            if i < len(self._hashkey):
                o2 = other._hashkey[i]
            if o1 is None:
                return True
            if o2 is None:
                return False
            if type(o1) != type(o2):
                return type(o1).__name__ < type(o2).__name__
            if isinstance(o1, Mapping):
                o1_items = sorted(o1.items())
                o2_items = sorted(o2.items())
                for o1_item, o2_item in zip(o1_items, o2_items):
                    o1_key, o1_value = o1_item
                    o2_key, o2_value = o2_item
                    if o1_key < o2_key:
                        return True
                    if o1_key > o2_key:
                        return False
                    if o1_value < o2_value:
                        return True
                    if o1_value > o2_value:
                        return False
            else:
                try:
                    if o1 > o2:
                        return False
                    if o1 < o2:
                        return True
                except TypeError:
                    return False
        return False

    def __gt__(self, other):
        for i in range(max(len(self._hashkey), len(other._hashkey))):
            o1, o2 = None, None
            if i < len(self._hashkey):
                o1 = self._hashkey[i]
            if i < len(self._hashkey):
                o2 = other._hashkey[i]
            if o2 is None:
                return True
            if o1 is None:
                return False
            if type(o1) != type(o2):
                return type(o1).__name__ > type(o2).__name__
            if isinstance(o1, Mapping):
                o1_items = sorted(o1.items())
                o2_items = sorted(o2.items())
                for o1_item, o2_item in zip(o1_items, o2_items):
                    o1_key, o1_value = o1_item
                    o2_key, o2_value = o2_item
                    if o1_key > o2_key:
                        return True
                    if o1_key < o2_key:
                        return False
                    if o1_value > o2_value:
                        return True
                    if o1_value < o2_value:
                        return False
            else:
                try:
                    if o1 < o2:
                        return False
                    if o1 > o2:
                        return True
                except TypeError:
                    return False
        return False

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

