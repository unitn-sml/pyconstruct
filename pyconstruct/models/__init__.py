"""\
Pyconstructi abstracts predictive models using a `Model` class. A `Model` is an
object that holds some kind of parameters that can be used to make inference.

When making inference, the model's parameters are passed to the Domain, which
needs to be able to interpret the semantics of the parameters. For instance, the
standard model provided by Pyconstruct is a `LinearModel`, whose parameters
consist in just a weight vector. Domains using the `solve` macro from the
`pyconstruct.pmzn` file (see the `MiniZincDomain` class documentation for
details) are readily capable of interpreting this weight vector and use it to
compute the dot product with the feature vector.

While linear models cover many of the cases in structured-output prediction, one
can easily think of cases in which an ad-hoc `Model` may be beneficial, e.g.
when the model has some hyper-paramenters or when performing some sort of
feature learning. That said, in most cases you should not need to manipulate
`Models` directly.

The `Model` is the middle-men between a `Learner` and a `Domain`. The learner
fits a `Model`, which is in turn then passed to the `Domain` to perform
inference. Usually a `Learner` is only capable of learning one type of models,
while a `Domain` may usually be used with different `Models` (if properly
configured).
"""

from . import base
from .base import *

__all__ = base.__all__

