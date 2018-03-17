
import os
import pymzn
pymzn.templates.add_package('pyconstruct.domains', 'predefined')

import numpy as np

from .base import BaseDomain, InferenceError

from pkg_resources import resource_exists, resource_filename


__all__ = ['get_path', 'get_source', 'get_info', 'print_info', 'MiniZincDomain']


def get_path(domain):
    file_name = domain if domain.endswith('.pmzn') else domain + '.pmzn'
    module = '.'.join([__package__, 'predefined'])
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
    the output objects `y` include all the independent optimization variables,
    whereas the input examples `x` are all the unspecified dzn parameters.
    Features are assumed to be included in a MiniZinc array named as the value
    of the `feature_var` argument.

    When using this class out-of-the-box, the domain is assumed to be encoded
    into a PyMzn `pmzn` domain file. PyMzn allows to templatize MiniZinc models
    using `Jinja2 <http://jinja.pocoo.org/>`_. With Jinja2, it is possible to
    reuse the same domain (attributes, constraints and features) to solve
    different inference problems. The most basic PyMzn domain would look like
    this:

        {% from 'pyconstruct.pmzn' import n_features, features, domain, solve %}

        {% call n_features(10) %}

        {% call domain(problem) %}
            % Your domain definition, e.g.
            int: x;
            var 0 .. 10: y;
            constraint x + y <= 10;
        {% endcall %}

        {% call features(problem) %}
            % Your features definition, e.g.
            [
                x + y,
                x - y
                % ...
            ]
        {% endcall %}

        {{ solve(problem, model, discretize=True) }}


    This is a MiniZinc file containing bits of templating in Jinja2. All the
    logic for handling several standard inference problems are hidden inside the
    various calls to the macros imported from 'pyconstruct.pmzn'.  A user of the
    library should be able to define a valid domain by only specifing the input
    and output variables, the constraints, and the features of the objects in a
    file with the above structure. You can see some examples in the
    pyconstruct/domains/predefined folder.

    Templating with Jinja2 is a powerful way to encode a domain, however, if you
    need some even more customization logic for inference, or you simply do not
    want to use MiniZinc, you may subclass the BaseDomain class and provide a
    custom implementation. In principle, learning algorithms should be agnostic
    to the type of objects a domain predicts, as long as the features are
    encoded as a `numpy.ndarray`; hence, inheriting from the BaseDomain class
    allows to use any other convenient representation for the objects and any
    other inference oracle.

    Parameters
    ----------
    domain_file : str
        The name of the predefined domain or the path to the `pmzn` file.
    feature_var : str
        The name of the MiniZinc variable containing the feature array.
    kwargs
        Any other argument needed by the template engine. These are passed to
        all inference problems solved by this domain.
    """
    def __init__(
        self, domain_file, feature_var='phi', n_features_var='N_FEATURES',
        cache=None, n_jobs=1, **kwargs
    ):
        super().__init__(cache=cache, n_jobs=n_jobs)

        if os.path.exists(domain_file):
            self.domain_file = domain_file
        else:
            self.domain_file = get_path(domain_file)
        if not self.domain_file:
            raise ValueError('Domain file not found.')

        self.feature_var = feature_var
        self.n_features_var = n_features_var
        self.args = kwargs

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
        stream = pymzn.minizinc(
            self.domain_file, data={**x, **y},
            args={**self.args, **kwargs, 'problem': 'phi'},
            output_vars=[self.feature_var]
        )

        if len(stream) == 0:
            raise InferenceError('Inference returned no solution.')

        return np.array(stream[-1][self.feature_var])

    def _infer(self, x, *args, model=None, problem='map', **kwargs):
        y_true = None
        if len(args) >= 1:
            y_true = args[0]

        if problem == 'phi':
            assert y_true is not None
            return self._phi(x, y_true, **kwargs)

        if model is None:
            model = BaseModel()

        args = {
            **self.args, **kwargs, 'model': model.parameters, 'problem': problem
        }

        if y_true is not None:
            args['y_true'] = y_true

        stream = pymzn.minizinc(self.domain_file, data=x, args=args)

        if len(stream) == 0:
            raise InferenceError('Inference returned no solution.')

        return stream[-1]

    def __repr__(self):
        return 'MiniZincDomain({}, {})'.format(self.domain_file, self.args)

    def __str__(self):
        return repr(self)

