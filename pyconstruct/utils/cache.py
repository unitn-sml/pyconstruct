
from sklearn.externals import joblib

try:
    from cachetools.keys import _kwmark
except ImportError:
    _kwmark = (object(),)


__all__ = ['hashkey', 'typedkey']


def _flatten(item):
    flat = []
    if isinstance(item, (list, tuple)):
        flat += [_flatten(e) for e in item]
    elif isinstance(item, dict):
        flat += [_flatten(e) for e in sum(sorted(item.items()), _kwmark)]
    else:
        return (item,)
    return tuple(flat)


class HashableArgs:


    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self._hashkey = _flatten(self.args) + _flatten(self.kwargs)
        self._hashvalue = None

    def __eq__(self, other):
       if len(self._hashkey) != len(other._hashkey):
           return False

       for o1, o2 in zip(self._hashkey, other._hashkey):
           try:
               if o1 != o2:
                   return False
           except (TypeError, ValueError):
               return False
       return True

    def __lt__(self, other):
        for i in range(max(len(self._hashkey), len(other._hashkey))):
            o1, o2 = None, None
            if i < len(self._hashkey):
                o1 = self._hashkey[i]
            if i < len(other._hashkey):
                o2 = other._hashkey[i]
            if o1 is None:
                return True
            if o2 is None:
                return False
            if type(o1) != type(o2):
                return type(o1).__name__ < type(o2).__name__
            try:
                if o1 > o2:
                    return False
                if o1 < o2:
                    return True
            except (TypeError, ValueError):
                return False
        return False

    def __gt__(self, other):
        for i in range(max(len(self._hashkey), len(other._hashkey))):
            o1, o2 = None, None
            if i < len(self._hashkey):
                o1 = self._hashkey[i]
            if i < len(other._hashkey):
                o2 = other._hashkey[i]
            if o2 is None:
                return True
            if o1 is None:
                return False
            if type(o1) != type(o2):
                return type(o1).__name__ > type(o2).__name__
            try:
                if o1 < o2:
                    return False
                if o1 > o2:
                    return True
            except (TypeError, ValueError):
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

