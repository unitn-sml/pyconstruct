"""\
This package provides the base classes to import domains into Pyconstruct, as
well as several predefined domains. A domain in Pyconstruct is an object that is
capable of interpreting the structure of some input and output objects, and
solve inference problems with these objects. The class `BaseDomain` defines the
basic interface that a domain should satisfy. An implementation of `BaseDomain`
has to be able to perform inference through the `infer` method and to transform
pairs of input and output objects into feature vector with the method `phi`.

Input and output objects can literally be any Python object, its the domain's
job to interpret them and perform inference over them. Learning algorithms in
Pyconstruct are built to be agnostic to the type of objects they are dealing
with. The only things the learners care about are the feature vectors, which
should be returned as Numpy arrays by the method `phi`.

The default type of Domains that Pyconstruct deals with are `MiniZincDomains`,
i.e. domains encoded with the MiniZinc constraint programming language. When a
domain is encoded in MiniZinc, it can be imported into Pyconstruct as easily
as::

    from pyconstruct import Domain
    domain = Domain('domain.pmzn')

When using a `MiniZincDomain` (`Domain` is an alias for `MiniZincDomain`) input
and output objects are encoded as Python dictionaries. Each object has a number
of "attributes", i.e. key-value pairs representing properties of the object.
When doing inference, input attributes are translated into `dzn` assignments to
pass to MiniZinc. In a MiniZinc domain, input attributes are bound to unassigned
`dzn` parameters. Output attributes can be, instead, any optimization variable
(usually the independent ones). MiniZinc is very flexible, because with it we
can solve several different inference problems.

By default, Pyconstruct assumes that domains are written into a variant of
MiniZinc as defined by the PyMzn library. Essentially, PyMzn extends MiniZinc
allowing to embed some templating code inside MiniZinc files. The templating
code is processed by the Jinja2 library. This is a powerful way to include
conditional boilerplate code, depending on the inference problem we are solving.
This allows us to write the domain once in a single `.pmzn` file and use it to
solve different inference problems. Pyconstruct includes a `.pmzn` module
containing several useful "macros" that can be used inside a domain file. A
typical domain file looks like this::

    {% from 'pyconstruct.pmzn' import n_features, features, domain, solve %}

    {{ n_features(1) }}

    {% call domain(problem) %}
        int: some_input_variable;

        var int: some_output_variable;

        % Some constraint
        constraint some_output_variable <= some_input_variable;

        {% if problem == 'loss_augmented_map' %}
            int: true_input_variable = {{ y_true['some_input_variable']|dzn }};
        {% endif %}

        {% call features() %}
            % Some features
            some_input_variable * some_output_variable
        {% endcall %}

    {% endcall %}

    {% set loss %}
        % Some loss function
        true_input_variable - some_input_variable
    {% endset %}

    {{ solve(problem, model, loss=loss) }}

As you can see, here we make use of some of the features of Pyconstruct to
define the domain. The first line is used to import some macros, which are then
called in the rest of the file. First we call `n_features`, which compiles into::

    int: N_FEATURES = 1;
    set of int: FEATURES = 1 .. N_FEATURES;

The second call is then to the `domain` macro, which accepts a `problem` string.
By default, each time inference is called by Pyconstruct a variable `problem`
containing the current inference problem is set into the global scope of Jinja2,
so it can be used like in this context. Inside the call to the `domain` macro,
we define the actual domain, including input parameters, output variables and
constraints. Here also go the features, and possibly declaration of true
variables for "loss_augmente_map" problems (see below). The call to the
`feature` macro is a convenience that compiles into::

    array[FEATURES] of var float: phi = [
        % Some features
        some_input_variable * some_output_variable
    ];

The Jinja2 `set` statement is then used to assign the `loss` variable, which is
then passed to the `solve` macro, which take care to insert in the compiled code
a proper MiniZinc `solve` statement, depending on the `problem` to solve and the
`model`. The `model` variable is just like `problem`: it is passed by the
`MiniZincDomain` during inference and it contains the parameters of the `Model`
we want to make inference with. For instance, a `LinearModel` contains just one
parameter `w`, which is an array of weights (see section Models for details).


Standard inference problems
~~~~~~~~~~~~~~~~~~~~~~~~~~~

These are the basic inference problems that Pyconstruct expects the Domains to
be able to solve:

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


# Alias
Domain = MiniZincDomain


__all__ = base.__all__ + minizinc.__all__ + ['Domain']

