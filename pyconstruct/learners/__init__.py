"""\
This package contains several learning algorithms to be used in conjunction with
Pyconstruct domains. All the learners in Pyconstruct are agnostic to the type of
structured objects the data contains, thanks to the fact that the domain takes
care of making inference in a generic way.

Here we use the "equations" example, available in the `source code
<https://github.com/unitn-sml/pyconstruct>`_, to show how Pyconstruct learners
can be used.

Learners in Pyconstruct follow the same interface as Scikit-learn estimators.
Each learner needs first to be instanciated, passing a domain as argument to the
constructor. For instance::

    from pyconstruct import SSG, Domain

    eq = Domain('equations.pmzn')
    ssg = SSG(eq)

Here we used `SSG`, which is a learner using the stochastic subgradient method.
A learner must be initialized with either a domain or an existing model
containing a domain. If only a domain is passed, then the default model for the
learner is used (a `LinearModel` in the case of `SSG`). If a model is also
passed, it has to be compatible with the learner (`LinearModel` or compatible
subclasses for `SSG`).

The constructor of the learner usually accepts other hyper-parameters of the
algorithm as well. After being instantiated, the learner needs to be trained
with data that is compatible with the domain passed to the learner instance. In
this case, for instance, we can make use of data provided by Pyconstruct::

    from pyconstruct.datasets import load_equations
    equations = load_equations()

Most of the learners in Pyconstruct are online learners, i.e. they can partially
fit a model a mini-batch of examples at the time. This provides high flexibility
to the way models can be trained, and is indeed useful given that training a
very big model on structured data may require a lot of time and computational
resources. As in Scikit-learn, online learners implement the `partial_fit`
method, which takes a mini-batch of examples and uses it to update the model.
Pyconstruct has a convenient utility to separate data into mini-batches, which
in turn can then be used to train the model::

    from pyconstruct.utils import batches

    for X_b, Y_b in batches(equations.data, equations.target, size=50):
        ssg.partial_fit(X_b, Y_b)

This method of training is very flexible because it allows to, for instance,
validate the model while training::

    from pyconstruct.metrics import hamming

    for X_b, Y_b in batches(equations.data, equations.target, size=50):

        # Validate
        Y_pred = ssg.predict(X_b)
        loss = hamming(Y_pred, Y_b, key='sequence').mean()
        print('Training loss: {}'.format(loss))

        # Update
        ssg.partial_fit(X_b, Y_b)

Here the `hamming` function takes a parameter `key='sequence'` because that is
the name of the attribute in the equations data that need to be compared.

Learners also implement the `fit` method, which divides the data in batches and
the uses the `partial_fit`, as shown above.
"""

from .base import *
from .subgradient import *
from .perceptron import *
from .frankwolfe import *


__all__ = (
        base.__all__ + subgradient.__all__ + perceptron.__all__
      + frankwolfe.__all__
)

