
from .base import *
from .minizinc import *


# Alias
Domain = MiniZincDomain


__all__ = base.__all__ + minizinc.__all__ + ['Domain']

