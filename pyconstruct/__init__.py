"""\
Pyconstruct is a Python library for declarative, constrained, structured-output
prediction. When using Pyconstruct, the problem specification can be encoded in
MiniZinc, a high-level constraint programming language. This means that domain
knowledge can be declaratively included in the inference procedure as
constraints over the optimization variables.
"""
from .domains import Domain
from . import datasets
from . import models
from .learners import *
from . import metrics
from . import utils


__all__ = (
    ['Domain', 'datasets', 'domains', 'models', 'learners', 'metrics', 'utils']
    + learners.__all__
)


__version__ = '0.1.8'

