
import numpy as np

from time import monotonic as _time

from .base import BaseLearner
from ..utils import get_logger, asarrays
from ..models import LinearModel, BaseModel

from scipy.special import expit
from abc import ABC, abstractmethod


__all__ = ['BaseSSG', 'SSG', 'EG']


def _project_l2(w, radius=1.0):
    norm = np.linalg.norm(w)
    if norm <= radius:
        return w
    return (w / norm) * radius


def _project_simplex(v, radius=1.0):
    """Compute the Euclidean projection on a positive simplex.

    Assume all components of v are positive. This is a O(n log(n))
    implementation, but a O(n) exists (see [1]).

    Code from:
    https://gist.github.com/daien/1272551

    Algorithm from:
    Efficient Projections onto the .1-Ball for Learning in High Dimensions
    John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
    International Conference on Machine Learning (ICML 2008)
    http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf
    """
    n, = v.shape
    if v.sum() == radius and np.alltrue(v >= 0):
        return v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - radius))[0][-1]
    theta = float(cssv[rho] - radius) / rho
    w = (v - theta).clip(min=0)
    return w


def _project_l1(v, radius=1.0):
    """Compute the Euclidean projection on a L1-ball.

    Code from:
    https://gist.github.com/daien/1272551

    Algorithm from:
    Efficient Projections onto the .1-Ball for Learning in High Dimensions
    John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
    International Conference on Machine Learning (ICML 2008)
    http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf
    """
    n, = v.shape
    u = np.abs(v)
    if u.sum() <= radius:
        return v
    w = _project_simplex(u, radius)
    return w * np.sign(v)


class BaseSSG(BaseLearner, ABC):
    """Base implementation of a learner with the subgradient algorithm.

    This class implements the backbone of a learner that uses the Stochastic
    Subgradient descent method [1]_ for updating the weights of a linear model.
    The `partial_fit` method takes care of making inference and calling a
    gradient step for each example in the batch. Implementations may use
    different methods for making a gradient step.

    Parameters
    ----------
    domain : BaseDomain
        The domain of the data.
    inference : str in ['map', 'loss_augmented_map']
        Which type of inference to perform when learning.
    structured_loss : function (y, y) -> float
        The structured loss to compute on the objects.

    References
    ----------
    .. [1] Ratliff, Nathan D., J. Andrew Bagnell, and Martin A. Zinkevich.
        "(Online) Subgradient Methods for Structured Prediction." Artificial
        Intelligence and Statistics. 2007.
    """

    def __init__(
        self, domain=None, inference='loss_augmented_map', structured_loss=None
    ):
        super().__init__(domain=domain)
        self.inference = inference
        self.structured_loss = structured_loss

    @abstractmethod
    def _step(self, w, x, y_true, y_pred):
        """Returns updated w"""

    def partial_fit(self, X, Y, Y_pred=None):
        if not hasattr(self, 'w_'):
            self.w_ = None
        if not hasattr(self, 't_'):
            self.iter_ = 0

        self.iter_ += 1

        log = get_logger(__name__)

        # Inference
        if Y_pred is None:
            start = _time()
            Y_pred = self.predict(X, problem=self.inference)
            infer_time = _time() - start

        w = None
        if self.w_ is not None:
            w = np.copy(self.w_)

        log.debug('''\
            Iteration {self.iter_:>2d}, current weights
            w = {w}
        ''', locals())

        # Weight updates
        learn_times = []
        for x, y_true, y_pred in zip(X, Y, Y_pred):
            start = _time()
            w = self._step(w, x, y_true, y_pred)
            learn_time = _time() - start
            learn_times.append(learn_time)

            log.debug('''\
                Iteration {self.iter_:>2d}, weights update
                x       = {x}
                y_true  = {y_true}
                y_pred  = {y_pred}
                w       = {w}
            ''', locals())

        self.w_ = w
        self.model_ = LinearModel(self.domain, self.w_)
        return self

    # Alias
    fit = partial_fit

    def loss(self, X, Y_true, Y_pred):
        loss = super().loss(X, Y_true, Y_pred)
        if self.structured_loss is not None:
            loss += self.structured_loss(Y_true, Y_pred)
        return loss


class SSG(BaseSSG):
    """Learner implementing the standard subgradient algorithm.

    This learner performs the standard Stochastic Subgradient descent from [1]_.
    It also includes the options for:

     - Training with the Pegasos update scheme [2]_; simply set `alpha` greater
       than zero to regularize the model.
     - Project onto an L2 or an L1 ball of a given radius, the latter using the
       projection algorithm from [3]_.
     - Boost the model with the method from [4]_; simply use a different
       training loss that `hinge`.
     - Adaptive step size techniques and such (coming soon).

    Parameters
    ----------
    domain : BaseDomain
        The domain of the data.
    inference : str in ['map', 'loss_augmented_map']
        Which type of inference to perform when learning.
    alpha : float
        The regularization coefficient.
    train_loss : str in ['hinge', 'logistic', 'exponential']
        The training loss. The derivative of this loss is used to rescale the
        margin of the examples when making an update.
    projection : None or str in ['l1', 'l2']
        If None, no projection is applied, otherwise, if 'l1' or 'l2' are given,
        the weights are projected back onto an L1 or an L2 ball respectively.
    radius : float
        The radius of the ball onto which project the weights when using
        projection.
    eta0 : float
        The initial value of the learning rate.
    power_t : float
        The power of the iteration index when using an `invscaling`
        learning_rate.
    learning_rate : str in ['constant', 'optimal', 'invscaling']
        The learning rate strategy. The `constant` learning multiplies the
        updates for `eta0`; the `invscaling` divides the updates by the
        iteration number raised to the `power_t`; the `optimal` strategy finds
        the best rate depending on `alpha` and `train_loss` (similar to
        Scikit-learn's SGDRegressor `optimal` learning rate).
    structured_loss : function (y, y) -> float
        The structured loss to compute on the objects.

    References
    ----------
    .. [1] Ratliff, Nathan D., J. Andrew Bagnell, and Martin A. Zinkevich.
        "(Online) Subgradient Methods for Structured Prediction." Artificial
        Intelligence and Statistics. 2007.
    .. [2] Shalev-Shwartz, Shai, et al. "Pegasos: Primal estimated sub-gradient
        solver for svm." Mathematical programming 127.1 (2011): 3-30.
    .. [3] Efficient Projections onto the .1-Ball for Learning in High
        Dimensions John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar
        Chandra. International Conference on Machine Learning (ICML 2008)
        http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf
    .. [4] Parker, Charles, Alan Fern, and Prasad Tadepalli. "Gradient boosting
        for sequence alignment." Proceedings of the 21st national conference on
        Artificial intelligence-Volume 1. AAAI Press, 2006.
    """

    def __init__(
        self, *, domain=None, inference='loss_augmented_map', alpha=0.0001,
        train_loss='hinge', projection='l2', radius=1000.0, eta0=1.0,
        power_t=0.5, learning_rate='optimal', structured_loss=None, **kwargs
    ):
        super().__init__(
            domain=domain, inference=inference, structured_loss=structured_loss
        )
        self.alpha = alpha
        self.train_loss = train_loss
        self.projection = projection
        self.radius = radius
        self.eta0 = eta0
        self.power_t = power_t
        self.learning_rate = learning_rate

    def _init_w(self, shape):
        return np.zeros(shape, dtype=np.float64)

    @property
    def _init_t(self):
        typw = np.sqrt(1.0 / np.sqrt(self.alpha))
        initial_eta0 = typw / max(1.0, self._dloss(-typw))
        return 1.0 / (initial_eta0 * self.alpha)

    def _exp(self, x, psi):
        # cap at self.radius if update would have greater norm
        norm = np.linalg.norm(psi)
        if np.log(norm) + x >= np.log(self.radius):
            return 1.0
        else:
            return np.exp(x) / self.radius

    def _dloss(self, loss, psi=1.0):
        return {
            'hinge': lambda x: 1.0,
            'logistic': lambda x: expit(x),
            'exponential': lambda x: self._exp(x, psi),
        }[self.train_loss](loss)

    def _eta(self):
        return {
            'constant': lambda t: self.eta0,
            'optimal': lambda t: 1.0 / (self.alpha * (self._init_t + t - 1)),
            'invscaling': lambda t: self.eta0 / np.power(t, self.power_t)
        }[self.learning_rate](self.t_)

    def _step(self, w, x, y_true, y_pred):
        if not hasattr(self, 't_'):
            self.t_ = 0
        self.t_ += 1

        phi_y_pred = self.domain.phi(*asarrays(x, y_pred))[0]
        phi_y_true = self.domain.phi(*asarrays(x, y_true))[0]
        psi = phi_y_true - phi_y_pred

        if w is None:
            w = self._init_w(psi.shape[0])

        margin = w.dot(psi)
        loss = - margin
        if self.structured_loss is not None:
            loss += self.structured_loss(y_true, y_pred)

        if isinstance(self.learning_rate, str):
            eta = self._eta()
        else:
            eta = self.learning_rate(self)

        rho = self._dloss(loss, eta * psi)
        step = self.alpha * w - psi * rho
        w -= eta * step
        if self.projection == 'l2':
            w = _project_l2(w, self.radius)
        elif self.projection == 'l1':
            w = _project_l1(w, self.radius)
        return w


class EG(BaseSSG):
    """Learner implementing the Exponentiate Gradient algorithm.

    This learner uses multiplicative weight updates as in [1]_.

    Parameters
    ----------
    domain : BaseDomain
        The domain of the data.
    inference : str in ['map', 'loss_augmented_map']
        Which type of inference to perform when learning.
    alpha : float
        The regularization coefficient.
    train_loss : str in ['hinge', 'logistic', 'exponential']
        The training loss. The derivative of this loss is used to rescale the
        margin of the examples when making an update.
    radius : float
        This property is only used here to decide when to clip the exponential
        training loss.
    eta0 : float
        The initial value of the learning rate.
    power_t : float
        The power of the iteration index when using an `invscaling`
        learning_rate.
    learning_rate : str in ['constant', 'optimal', 'invscaling']
        The learning rate strategy. The `constant` learning multiplies the
        updates for `eta0`; the `invscaling` divides the updates by the
        iteration number raised to the `power_t`; the `optimal` strategy finds
        the best rate depending on `alpha` and `train_loss` (similar to
        Scikit-learn's SGDRegressor `optimal` learning rate).
    structured_loss : function (y, y) -> float
        The structured loss to compute on the objects.

    References
    ----------
    .. [1] Collins, Michael, et al. "Exponentiated gradient algorithms for
        conditional random fields and max-margin markov networks." Journal of
        Machine Learning Research 9.Aug (2008): 1775-1822.
    """

    def __init__(
        self, *, domain=None, inference='loss_augmented_map', alpha=0.0001,
        train_loss='hinge', radius=1000.0, eta0=1.0, power_t=0.5,
        learning_rate='optimal', structured_loss=None, **kwargs
    ):
        super().__init__(
            domain=domain, inference=inference, structured_loss=structured_loss
        )
        self.alpha = alpha
        self.train_loss = train_loss
        self.radius = radius
        self.eta0 = eta0
        self.power_t = power_t
        self.learning_rate = learning_rate

    def _init_w(dim):
        return np.full(dim, 1.0 / dim)

    @property
    def _init_t(self):
        typw = np.sqrt(1.0 / np.sqrt(self.alpha))
        initial_eta0 = typw / max(1.0, self._dloss(-typw))
        return 1.0 / (initial_eta0 * self.alpha)

    def _exp(self, x, psi):
        # cap at self.radius if update would have greater norm
        norm = np.linalg.norm(psi)
        if np.log(norm) + x >= np.log(self.radius):
            return 1.0
        else:
            return np.exp(x) / self.radius

    def _dloss(self, loss, psi=1.0):
        return {
            'hinge': lambda x: 1.0,
            'logistic': lambda x: expit(x),
            'exponential': lambda x: self._exp(x, psi),
        }[self.train_loss](loss)

    def _eta(self):
        return {
            'constant': lambda t: self.eta0,
            'optimal': lambda t: 1.0 / (self.alpha * (self._init_t + t - 1)),
            'invscaling': lambda t: self.eta0 / np.power(t, self.power_t)
        }[self.learning_rate](self.t_)

    def _step(self, w, x, y_true, y_pred):
        if not hasattr(self, 't_'):
            self.t_ = 0
        self.t_ += 1

        phi_y_pred = self.domain.phi(x, y_pred)
        phi_y_true = self.domain.phi(x, y_true)
        psi = phi_y_true - phi_y_pred

        if w is None:
            w = self._init_w(psi.shape[0])

        margin = w.dot(psi)
        loss = - margin
        if self.structured_loss is not None:
            loss += self.structured_loss(y_true, y_pred)

        if isinstance(self.learning_rate, str):
            eta = self._eta()
        else:
            eta = self.learning_rate(self)

        rho = self._dloss(loss, eta * psi)
        step = self.alpha * w - psi * rho
        w *= np.exp(- eta * step)
        w /= np.sum(w)
        return w

