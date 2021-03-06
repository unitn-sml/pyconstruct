
import os
import pymzn
import pymzn.templates
pymzn.templates.add_package('pyconstruct.domains', 'share')

import numpy as np

from .base import BaseDomain, InferenceError
from ..utils import dictsplit, hashkey

from pkg_resources import resource_exists, resource_filename


__all__ = [
    'get_path', 'get_source', 'get_info', 'print_info', 'MiniZincDomain'
]


def get_path(domain):
    file_name = domain if domain.endswith('.pmzn') else domain + '.pmzn'
    module = '.'.join([__package__, 'share'])
    if resource_exists(module, file_name):
        return resource_filename(module, file_name)
    return None


def get_source(domain):
    path = get_path(domain)
    with open(path) as f:
        return f.read()


def get_info(domain):
    path = get_path(domain)
    info = []
    with open(path) as f:
        line = next(f)
        while line.startswith('%'):
            info.append(line[1:])
            line = next(f)
    return ''.join(info)


def print_info(domain):
    print(get_info(domain))


class MiniZincDomain(BaseDomain):
    """Base class for domains encoded in minizinc.

    This is the default method for encoding a domain in Pyconstruct. When using
    MiniZinc to encode a domain, objects are represented in terms of their
    attributes, i.e. as dictionaries containing attribute names associated to
    their values. The default inference oracle is assumed to be
    `MiniZinc <http://www.minizinc.org/>`_, through its Python interface `PyMzn
    <https://github.com/paolodragone/pymzn>`_. Using MiniZinc, the attributes of
    the output objects `y` include all independent optimization variables.
    Input examples `x` are also dictionaries containing attributes corresponding
    to all the unspecified dzn parameters in the minizinc file.

    When using this class out-of-the-box, the domain is assumed to be encoded
    into a PyMzn `pmzn` domain file. PyMzn allows to templatize MiniZinc models
    using `Jinja2 <http://jinja.pocoo.org/>`_. With Jinja2, it is possible to
    reuse the same domain (attributes, constraints and features) to solve
    different inference problems. The most basic PyMzn domain would look like
    this::

        {% from 'globals.pmzn' import domain, solve %}
        {% from 'linear.pmzn' import linear_model %}

        {% call domain(problem) %}
            % Your domain definition, e.g.
            int: x;
            var 0 .. 10: y;
            constraint x + y <= 10;
        {% endcall %}

        {% call linear_model(problem, params, n_features='10') %}
            % Your features definition, e.g.
            [
                x + y,
                x - y
                % ...
            ]
        {% endcall %}

        {{ solve(problem) }}


    This is a MiniZinc file containing bits of templating in Jinja2. All the
    logic for handling several standard inference problems are hidden inside the
    various calls to the macros imported from `linear.pmzn` and `globals.pmzn`.
    A user of the library should be able to define a valid domain by only
    specifing the input and output variables, the constraints, and the features
    of the objects in a file with the above structure. You can see some examples
    in the pyconstruct/examples folder in the `source code
    <https://github.com/unitn-sml/pyconstruct>`_.

    Templating with Jinja2 is a powerful way to encode a domain, however, if you
    need some even more customization logic for inference, or you simply do not
    want to use MiniZinc, you may subclass the BaseDomain class and provide a
    custom implementation. In principle, learning algorithms should be agnostic
    to the type of objects a domain returns, as long as the features are encoded
    as a `numpy.ndarray` (in case of a linear model); hence, inheriting from the
    BaseDomain class allows to use any other convenient representation for the
    objects and any other inference oracle.

    The shared templating library of Pyconstruct is organized in such a way to
    group together helper macros pertaining the same kind of models. For
    instance, the file `linear.pmzn` contains helper macros that can be used in
    conjunction with LinearModels. Some files like `globals.pmzn` or
    `metrics.pmzn` contain instead routines useful for all kinds of models.

    Currently, linear models are the only kind supported by Pyconstruct, but we
    plan to provide more, e.g. CRFs.

    Parameters
    ----------
    domain_file : str
        The path to the `pmzn` file.
    feature_var : str
        The name of the MiniZinc variable containing the feature array. Used for
        getting the feature vector when the domain encodes inference over a
        linear model.
    n_feature_var : str
        The name of the MiniZinc variable containing the number of features.
        Used for getting the number of features when the domain encodes
        inference over a linear model.
    timeout : None or int
        The timeout to give to the solver (in seconds). None (default) means no
        timeout.
    kwargs
        Any other argument needed by the template engine. These are passed to
        all inference problems solved by the domain.
    """
    def __init__(
        self, domain_file, feature_var='phi', n_features_var='N_FEATURES',
        cache=None, n_jobs=1, timeout=None, **kwargs
    ):
        super().__init__(cache=cache, n_jobs=n_jobs)

        if os.path.exists(domain_file):
            self.domain_file = domain_file
        if not hasattr(self, 'domain_file'):
            raise ValueError('Domain file not found.')

        self.feature_var = feature_var
        self.n_features_var = n_features_var
        self.timeout = timeout
        self.args = kwargs
        self._x_vars = None
        self._y_vars = None

    def n_features(self, **kwargs):
        """Return the number of features in the feature vector.

        Parameters
        ----------
        kwargs
            Additional arguments needed by the domain file.
        """
        args = {'problem': 'n_features', **self.args, **kwargs}
        stream = pymzn.minizinc(
            self.domain_file, problem='n_features', args=args,
            output_vars=[self.n_features_var]
        )

        if len(stream) == 0 or self.n_features_var not in stream[-1]:
            raise InferenceError('Problem n_features not supported.')

        return stream[-1][self.n_features_var]

    def _phi(self, x, y, **kwargs):
        if self._x_vars is None:
            self._x_vars = list(x.keys())
        if self._y_vars is None:
            self._y_vars = list(y.keys())

        stream = pymzn.minizinc(
            self.domain_file, data={**x, **y},
            args={**self.args, **kwargs, 'problem': 'phi'},
            output_vars=[self.feature_var]
        )

        if len(stream) == 0:
            raise InferenceError('Inference returned no solution.')

        return np.array(stream[-1][self.feature_var])

    def _infer(
        self, x, *args, model=None, problem='map', return_phi=False, **kwargs
    ):
        if self._x_vars is None:
            self._x_vars = list(x.keys())

        y_true = None
        if len(args) >= 1:
            y_true = args[0]
            if self._y_vars is None:
                self._y_vars = list(y_true.keys())

        if problem == 'phi':
            assert y_true is not None
            return self._phi(x, y_true, **kwargs)

        if model is None:
            model = BaseModel(self)

        args = {
            **self.args, **kwargs, 'params': model.params, 'problem': problem
        }

        output_vars = None
        if problem and self._y_vars:
            output_vars = self._y_vars + [self.feature_var]

        if y_true is not None:
            args['y_true'] = y_true

        _timeout = self.timeout
        stream = None
        while not stream:
            try:
                stream = pymzn.minizinc(
                    self.domain_file, data=x, args=args,
                    output_vars=output_vars, timeout=_timeout
                )
            except pymzn.MiniZincError:
                if _timeout is not None:
                    _timeout *= 2
                else:
                    raise

        if len(stream) == 0:
            raise InferenceError('Inference returned no solution.')

        y, res = dictsplit(stream[-1], self._y_vars)

        if return_phi:
            phi = None
            if self.feature_var in res:
                phi = np.array(res[self.feature_var])
            else:
                phi = self._phi(x, y, **kwargs)
            return y, phi
        return y

    def __repr__(self):
        return 'MiniZincDomain({}, {})'.format(self.domain_file, self.args)

    def __str__(self):
        return repr(self)

