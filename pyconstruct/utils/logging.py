
import logging
import numpy as np

from textwrap import dedent
from collections.abc import Mapping

from .arrays import array2str
from .dicts import dict2str, dictsplit


__all__ = ['get_logger']


class BraceMessage:
    """A logging message using braces formatting.

    Parameters
    ----------
    fmt : str
        The format string.
    args : list
        The argument list. If the list contains only one dictionary, it is
        used to get keyword arguments instead.
    kwargs : dict
        The keyword arguments dictionary.
    """
    def __init__(self, fmt, args, kwargs):
        self.fmt = fmt
        self.kwargs = kwargs
        if args and len(args) == 1 and isinstance(args[0], Mapping) and args[0]:
            self.args = ()
            self.kwargs = {**args[0], **self.kwargs}
        else:
            self.args = args

    @staticmethod
    def _eval(arg):
        if isinstance(arg, np.ndarray):
            return array2str(arg)
        elif isinstance(arg, dict):
            return dict2str(arg)
        return arg

    def __str__(self):
        args = [self._eval(arg) for arg in self.args]
        kwargs = {key: self._eval(arg) for key, arg in self.kwargs.items()}
        return self.fmt.format(*args, **kwargs)


class BracesAdapter(logging.LoggerAdapter):
    """An adapter to allow braces formatting for logging messages."""

    def __init__(self, logger, extra=None):
        super().__init__(logger, extra or {})

    def log(self, level, msg, *args, **kwargs):
        if self.isEnabledFor(level):
            logargs, msgargs = dictsplit(
                kwargs, ['exc_info', 'extra', 'stack_info']
            )
            msg = BraceMessage(dedent(str(msg)), args, msgargs)
            self.logger._log(level, msg, (), **logargs)


def get_logger(name):
    """Utility to get a logger with a BracesAdapter"""
    return BracesAdapter(logging.getLogger(name))

