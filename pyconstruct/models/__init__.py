"""\
Pyconstructi abstracts predictive models using the `BaseModel` class. Classes
that inherit from `BaseModel` are objects holding the parameters of some
mathematical model and that can "communicate" with domains in order to make
prediction by solving inference problems.

When making inference, the model's parameters are passed to the Domain, which
needs to be able to interpret the semantics of the parameters. For instance, the
standard model provided by Pyconstruct is a `LinearModel`, whose parameters
consist in just a weight vector (and some optional additional features). Domains
using the `linear_model` macro from the `linear.pmzn` file (see the `domains`
module documentation for details) are readily capable of interpreting this
weight vector and use it to compute the dot product with the feature vector.

While linear models cover many of the cases in structured-output prediction, one
can easily think of cases in which an ad-hoc model may be beneficial, e.g.
when the model has some hyper-paramenters or when performing some sort of
feature learning. That said, in most cases you should not need to manipulate
models directly.

The model is the middle-men between a learner and a domain. The learner fits
the parameter of a model, without knowing how inference is carried out. The
domain, on the other hand, receives the model's parameters, without knowing how
these are learned. Usually a learner is only capable of learning one type of
models (and derived ones), while a domain may conditionally be used with
different models (if properly configured).
"""

from . import base
from .base import *

__all__ = base.__all__

