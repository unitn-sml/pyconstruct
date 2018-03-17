
from . import base
from .base import *
from . import minizinc
from .minizinc import *


# Alias
Domain = MiniZincDomain


__all__ = base.__all__ + minizinc.__all__ + ['Domain']

