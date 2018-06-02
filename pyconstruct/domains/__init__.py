"""\
This package provides the base classes to import domains into Pyconstruct, as
well as the Pyconstruct shared templating library. A domain in Pyconstruct is an
object that is capable of interpreting the structure of some input and output
objects, and solve inference problems with these objects. The class `BaseDomain`
defines the basic interface that a domain should satisfy. An implementation of
`BaseDomain` has to be able to perform inference through the `infer` method.

Input and output objects can literally be any Python object, its the domain's
job to interpret them and perform inference over them. Learning algorithms in
Pyconstruct are built to be agnostic to the type of objects they are dealing
with.

The default type of Domains that Pyconstruct provides are `MiniZincDomains`,
i.e. domains encoded with the MiniZinc constraint programming language. When a
domain is encoded in MiniZinc, it can be imported into Pyconstruct as easily
as::

    from pyconstruct import Domain
    domain = Domain('domain.pmzn')

When using a `MiniZincDomain`, input and output objects are encoded as Python
dictionaries. Each object has a number of "attributes", i.e. key-value pairs
representing properties of the object.  When doing inference, input attributes
are translated into `dzn` assignments to pass to MiniZinc. In a MiniZinc domain,
input attributes are bound to unassigned `dzn` parameters. Output attributes can
be, instead, any optimization variable (usually the independent ones). MiniZinc
is very flexible, as it allows us to solve several different inference problems
using the same framework.

By default, Pyconstruct assumes that domains are written into a variant of
MiniZinc defined by the PyMzn library. Essentially, PyMzn extends MiniZinc
allowing to embed some templating code inside MiniZinc files. The templating
code is processed by the Jinja2 library. This is a powerful way to include
conditional boilerplate code, depending on the inference problem we are solving.
This allows us to write the domain once in a single `.pmzn` file and use it to
solve different inference problems. Pyconstruct includes a shared library of
`.pmzn` modules, containing several helper "macros" that can be used inside a
domain file. A typical domain file looks like this::

    {% from 'globals.pmzn' import domain, solve %}
    {% from 'linear.pmzn' import linear_model %}

    {% call domain(problem) %}
        int: some_input_variable;
        var int: some_output_variable;

        % Some constraint
        constraint some_output_variable <= some_input_variable;

        {% if problem == 'loss_augmented_map' %}
            int: true_output_variable = {{ y_true['some_output_variable']|dzn }};
        {% endif %}
    {% endcall %}

    {% call linear_model(problem, params, n_features='1') %}
        % Some features
        some_input_variable * some_output_variable
    {% endcall %}

    {% set loss %}
        % Some loss function
        abs(true_output_variable - some_output_variable)
    {% endset %}

    {{ solve(problem, loss=loss) }}

As you can see, here we make use of some of the macros in the Pyconstruct shared
templating library to define the domain. The first lines are used to import some
macros, which are then used in the rest of the file.

The first call is to the `domain` macro, which accepts a `problem` string.  By
default, each time inference is called by Pyconstruct, a variable `problem`
containing the current inference problem is set into the global scope of Jinja2,
so it can be used like in this context. Inside the call to the `domain` macro,
we define the actual domain, including input parameters, output variables and
constraints. Here are also defined the true variables for "loss_augmente_map"
problems (see below).

The call to the `linear_model` macro is a convenience that compiles into::

    int: N_FEATURES = 1;
    set of int: FEATURES = 1 .. N_FEATURES;
    array[FEATURES] of var float: phi = [
        % Some features
        some_input_variable * some_output_variable
    ];
    array[FEATURES] of float: w = [ ... ];
    var float: score = sum(i in FEATURES)(w[i] * phi[i]);

The weights `w` are taken from the parameters of the model contained in the
global variable `params`, which is passed to the `linear_model` call.

The Jinja2 `set` statement is used to assign the `loss` variable, which is then
passed to the `solve` macro, which take care to insert in the compiled code a
proper MiniZinc `solve` statement, depending on the `problem` to solve.


Standard inference problems
~~~~~~~~~~~~~~~~~~~~~~~~~~~

These are the basic inference problems that Pyconstruct expects the Domains to
be able to solve (in order to work with LinearModels):

 - **n_features** : Returns the number of features in the feature vector;
 - **phi** : Given a `x` and `y`, returns the feature vector `phi(x, y)`;
 - **map** : Given a `x` and a `model`, returns the `y` that maximizes the score
   according to the given `model` and input `x`;
 - **loss_augmented_map** : Given a `x`, a `model` and a `y_true`, find `y` that
   maximizes the score + loss, according to the given `model`, input `x` and true
   output `y_true`.

"""

from .base import *
from .minizinc import *
from .share import *


def Domain(domain, **kwargs):
    """Meta-function for instantiating a domain from different sources"""
    if domain.endswith('.pmzn'):
        return MiniZincDomain(domain, **kwargs)
    raise ValueError('domain not recognized')


__all__ = base.__all__ + minizinc.__all__ + ['Domain', 'share']

