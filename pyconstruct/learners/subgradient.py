
import numpy as np

from time import monotonic as _time

from .base import BaseLearner
from ..utils import get_logger, asarrays, broadcast, hashkey, batches
from ..models import LinearModel, BaseModel

from scipy.special import expit
from abc import ABC, abstractmethod

from sklearn.utils import check_random_state, shuffle


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
    alpha : float
        The regularization coefficient.
    train_loss : str in ['hinge', 'logistic', 'exponential']
        The training loss. The derivative of this loss is used to rescale the
        margin of the examples when making an update.
    structured_loss : function (y, y) -> float
        The structured loss to compute on the objects.
    n_jobs : int
        The number of parallel jobs used when calculating the gradient steps.
    shuffle : bool
        Wheter to shuffle the data prior to fitting.
    warm_start : bool
        If True, the model is not reinitialized upon a second call to the fit
        method, and the parameters of the fit method are reused.
    verbose : bool
        Whether to print the training losses when fitting.
    batch_size : None or int
        The size of the batches used when fitting the data. If None, the batch
        size will be equal to `n_jobs`.
    validate : bool
        Whether to validate the model using the training examples prior to the
        update. Only used when `verbose` is True.
    random_state : None, int, RandomState
        The random number generator seed.

    References
    ----------
    .. [1] Ratliff, Nathan D., J. Andrew Bagnell, and Martin A. Zinkevich.
        "(Online) Subgradient Methods for Structured Prediction." Artificial
        Intelligence and Statistics. 2007.
    """

    def __init__(
        self, domain=None, inference='loss_augmented_map', alpha=0.0001,
        train_loss='hinge', structured_loss=None, n_jobs=1, shuffle=True,
        warm_start=False, verbose=True, batch_size=None, validate=True,
        random_state=None, **kwargs
    ):
        super().__init__(domain)
        self.inference = inference
        self.alpha = alpha
        self.train_loss = train_loss
        self.structured_loss = structured_loss
        self.n_jobs = n_jobs
        self.shuffle = shuffle
        self.warm_start = warm_start
        self.verbose = verbose
        self.batch_size = batch_size
        self.validate = validate
        self.random_state = check_random_state(random_state)

    @abstractmethod
    def _learning_rate(self):
        """Returns the current learning rate"""

    @abstractmethod
    def _step(self, x, y_true, y_pred, phi_y_true, phi_y_pred, w, eta):
        """Returns a gradient step"""

    @abstractmethod
    def _update(self, w, step, eta):
        """Returns updated w"""

    def _exp(self, x, psi):
        # cap at self.radius_ if update would have greater norm
        norm = np.linalg.norm(psi)
        if np.log(norm) + x >= np.log(self.radius_):
            return 1.0
        else:
            return np.exp(x) / self.radius_

    def _dloss(self, loss, psi=1.0):
        return {
            'hinge': lambda x: 1.0,
            'logistic': lambda x: expit(x),
            'exponential': lambda x: self._exp(x, psi),
        }[self.train_loss](loss)

    def _update_phi_cache(self, X, Y, Y_phi, **kwargs):
        if self.domain.cache is not None:
            for x, y, y_phi in zip(X, Y, Y_phi):
                key = hashkey(x, y, problem='phi', **kwargs)
                if key not in self.domain.cache:
                    self.domain.cache[key] = y_phi

    def partial_fit(self, X, Y, Y_pred=None, Y_phi=None, Y_pred_phi=None):
        if not hasattr(self, 'w_'):
            self.w_ = None
        if not hasattr(self, 't_'):
            self.t_ = 0
        self.t_ += 1

        # Inference
        if Y_pred is None:
            start = _time()
            Y_pred, Y_pred_phi = zip(*self.predict(
                X, Y, return_phi=True, problem=self.inference
            ))
            infer_time = _time() - start
            self._update_phi_cache(X, Y_pred, Y_pred_phi)

        if Y_phi is None:
            Y_phi = self.phi(X, Y)
            self._update_phi_cache(X, Y, Y_phi)

        if Y_pred_phi is None:
            Y_pred_phi = self.phi(X, Y_pred)
            self._update_phi_cache(X, Y_pred, Y_pred_phi)

        if self.w_ is not None:
            w = np.copy(self.w_)
        else:
            w = self._init_w(Y_phi[-1].shape[0])

        eta = self._learning_rate()

        # Weight updates
        steps = broadcast(
            self._step, X, Y, Y_pred, Y_phi, Y_pred_phi, w=w, eta=eta,
            n_jobs=self.n_jobs
        )

        self.w_ = self._update(w, steps.mean(axis=0), eta)
        self.model_ = LinearModel(self.domain, self.w_)
        return self

    def fit(self, X, Y):
        if not self.warm_start:
            self.w_ = None
            self.t_ = 0

        self.n_samples_ = X.shape[0]

        if self.shuffle:
            X, Y = shuffle(X, Y, random_state=self.random_state)

        losses = np.array([])
        batch_size = self.batch_size or self.n_jobs

        for n_b, (X_b, Y_b) in enumerate(batches(X, Y, batch_size=batch_size)):
            if self.verbose:
                print('BATCH N. {}'.format(n_b + 1))
                print('\tSamples: {}'.format(batch_size*n_b + X_b.shape[0]))

                if self.validate and self.structured_loss is not None:
                    Y_pred = self.predict(X_b)
                    loss = self.structured_loss(Y_b, Y_pred).mean()
                    losses = np.append(losses, loss)
                    avg_loss = losses.sum() / (losses.size)
                    print('\tTraining loss: {:>5.4f}'.format(loss))
                    print('\tAverage training loss: {:>5.4f}'.format(avg_loss))
            self.partial_fit(X_b, Y_b)

        return self

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
    structured_loss : function (y, y) -> float
        The structured loss to compute on the objects.
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
    radius : float
        The radius of the ball enclosing the parameter space.
    projection : None or str in ['l1', 'l2']
        If None, no projection is applied, otherwise, if 'l1' or 'l2' are given,
        the weights are projected back onto an L1 or an L2 ball respectively.
    init_w : str in ['zeros', 'uniform', 'normal', 'laplace']
        Initialization strategy for the parameter vector. 'zeros' initializes
        the vector to all zero values; 'uniform', 'normal' and 'laplace'
        initialize the vector with random weights from, respectively, a uniform,
        normal, or Laplace distribution.

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
        self, domain=None, inference='loss_augmented_map', eta0=1.0,
        power_t=0.5, learning_rate='optimal', radius=1000.0, projection='l2',
        init_w='normal', **kwargs
    ):
        super().__init__(domain, inference=inference, **kwargs)
        self.eta0 = eta0
        self.power_t = power_t
        self.learning_rate = learning_rate
        self.radius = self.radius_ = radius
        self.projection = projection
        self.init_w = init_w

    def _init_w(self, shape):
        w = {
            'zeros': lambda s: np.zeros(s, dtype=np.float64),
            'uniform': lambda s: self.random_state.uniform(0.0, 1.0, s),
            'normal': lambda s: self.random_state.normal(0.0, 1.0, s),
            'laplace': lambda s: self.random_state.laplace(0.0, 1.0, s)
        }[self.init_w](shape)
        norm = np.linalg.norm(w)
        if norm > 0:
            return w / norm
        return w

    @property
    def _init_t(self):
        if self.alpha > 0:
            typw = np.sqrt(1.0 / np.sqrt(self.alpha))
            initial_eta0 = typw / max(1.0, self._dloss(-typw))
            return 1.0 / (initial_eta0 * self.alpha)
        return 1.0

    def _learning_rate(self):
        return {
            'constant': lambda t: self.eta0,
            'optimal': lambda t: 1.0 / (self.alpha * (self._init_t + t - 1.0)),
            'invscaling': lambda t: self.eta0 / np.power(t, self.power_t)
        }[self.learning_rate](self.t_)

    def _step(self, x, y_true, y_pred, phi_y_true, phi_y_pred, w, eta):
        psi = phi_y_true - phi_y_pred
        margin = w.dot(psi)
        loss = - margin
        if self.structured_loss is not None:
            loss += self.structured_loss(*asarrays(y_true, y_pred))

        rho = self._dloss(loss, eta * psi)
        step = self.alpha * w - psi * rho
        return step

    def _update(self, w, step, eta):
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
    structured_loss : function (y, y) -> float
        The structured loss to compute on the objects.
    eta0 : float
        The initial value of the learning rate.
    power_t : float
        The power of the iteration index when using an `invscaling`
        learning_rate.
    learning_rate : str in ['constant', 'decaying', 'invscaling']
        The learning rate strategy. The `constant` learning multiplies the
        updates for `eta0`; the `invscaling` divides the updates by the
        iteration number raised to the `power_t`; the `decaying` strategy
        decreases monotonically within the range [0.5, 1] with the number of
        samples seen. Same strategy used in [1]_.
    n_samples : int
        Estimate of the number of samples in the dataset. This parameter helps
        setting the decaying learning rate when training is initialized with the
        `partial_fit` instead of the `fit` method.

    References
    ----------
    .. [1] Collins, Michael, et al. "Exponentiated gradient algorithms for
        conditional random fields and max-margin markov networks." Journal of
        Machine Learning Research 9.Aug (2008): 1775-1822.
    """

    def __init__(
        self, domain=None, inference='map', eta0=1.0, power_t=0.5,
        learning_rate='decaying', n_samples=1000, **kwargs
    ):
        super().__init__(domain, inference=inference, **kwargs)
        self.eta0 = eta0
        self.power_t = power_t
        self.learning_rate = learning_rate
        self.n_samples = self.n_samples_ = n_samples

    def _init_w(dim):
        return np.full(dim, 1.0 / dim)

    def _learning_rate(self):
        return {
            'constant': lambda t: self.eta0,
            'decaying': lambda t: self.eta0 / (1.0 + self.t_ / self.n_samples_),
            'invscaling': lambda t: self.eta0 / np.power(t, self.power_t)
        }[self.learning_rate](self.t_)

    def _step(self, x, y_true, y_pred, phi_y_true, phi_y_pred, w, eta):
        if not hasattr(self, 'radius_'):
            self.radius_ = w.shape[0]

        psi = phi_y_true - phi_y_pred
        margin = w.dot(psi)
        loss = - margin
        if self.structured_loss is not None:
            loss += self.structured_loss(y_true, y_pred)

        rho = self._dloss(loss, eta * psi)
        step = self.alpha * w - psi * rho
        return step

    def _update(self, w, step, eta):
        w *= np.exp(- eta * step)
        w /= np.sum(w)
        return w

