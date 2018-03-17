
import re
import inspect


__all__ = [
    'get_class', 'get_defaults', 'add_prefix', 'strip_prefix', 'parse_remainder'
]


def get_class(name, defmod=None):
    """Finds a class.

    Search a class from its fully qualified name, e.g.
    'pyconstruct.domains.predefined.Class'.  If the class name is not fully
    qualified, e.g. 'Class', it is searched inside the default module.

    Parameter
    ---------
    name : str
        The fully qualified name of the class or the name of the class or the
        class name in the default module.
    defmod : str
        The default module where to search the class if not fully qualified.
    """
    ns = name.split('.')
    if len(ns) == 1:
        if not defmod:
            raise ValueError(
                'Provide either the fully qualified name of the class or the '
                'default module where to look for it'
            )
        mod = defmod
    else:
        mod = '.'.join(ns[:-1])
    module = __import__(mod, fromlist=[ns[-1]])
    return getattr(module, ns[-1])


def get_defaults(func):
    """Gets the default values of the keyword arguments a function.

    Parameter
    ---------
    func : function
        A function.
    """
    spec = inspect.getfullargspec(func)
    defaults = {}
    if spec.defaults:
        defs = spec.defaults
        args = spec.args
        for i in range(len(defs)):
            defaults[args[i + len(args) - len(defs)]] = defs[i]
    if spec.kwonlydefaults:
        defaults = {**defaults, **spec.kwonlydefaults}
    return defaults


def add_prefix(attrs, lst, prefix):
    rlst = []
    for s in lst:
        for attr in attrs:
            s = re.sub(r'\b{}\b'.format(attr), prefix + attr, s)
        rlst.append(s)
    return rlst


def strip_prefix(y, prefix):
    """Return y without the prefix."""
    return {k[len(prefix):]: v for k, v in y.items() if k.startswith(prefix)}


def parse_remainder(args):
    """Parse the remainder of the command line arguments"""
    if len(args) == 0:
        return {}
    if not args[0].startswith(('-', '--')):
        raise ValueError('Only optional arguments allowed')

    kwargs = {}
    curr_arg = args[0]
    curr_val = None
    for arg in args[1:]:
        if arg.startswith(('-', '--')):
            if not curr_val:
                kwargs[curr_arg] = True
            else:
                kwargs[curr_arg] = curr_val
            curr_arg = arg
            curr_val = None
        else:
            if not curr_val:
                cur_val = arg
            elif isinstance(cur_val, list):
                cur_val.append(arg)
            else:
                cur_val = [cur_val]
                cur_val.append(arg)
    return kwargs

