
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
    [1] Efficient Projections onto the .1-Ball for Learning in High Dimensions
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
    [1] Efficient Projections onto the .1-Ball for Learning in High Dimensions
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
            'optimal': lambda t: 1.0 / (alpha * (self._init_t + t - 1)),
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
    """Exponentiated gradient algorithm."""

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
            'optimal': lambda t: 1.0 / (alpha * (self._init_t + t - 1)),
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

