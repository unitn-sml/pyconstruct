
import numpy as np

from .base import BaseLearner
from ..models import LinearModel
from ..utils import get_logger, hashkey, asarrays

from time import monotonic as _time


__all__ = ['BlockCoordinateFrankWolfe']


class BlockCoordinateFrankWolfe(BaseLearner):
    """Learner using the Block-Coordinate Frank-Wolfe algorithm [1]_.

    This implementation is still a bit experimental. Should work fine but still
    has not been made parallel, so it will take much more than necessary on most
    datasets. Coming soon also improvements from [2]_.

    Parameters
    ----------
    domain : BaseDomain
        The domain of the data.
    structured_loss : function (y, y) -> float
        The structured loss to compute on the objects.
    dataset_size : int
        A hint on the size of the dataset. May improve performance.
    alpha : float
        The regularization coefficient.

    References
    ----------
    .. [1] Lacoste-Julien, Simon, et al. "Block-Coordinate Frank-Wolfe
        Optimization for Structural SVMs." ICML 2013 International Conference
        on Machine Learning. 2013.
    .. [2] Osokin, Anton, et al. "Minding the Gaps for Block Frank-Wolfe
        Optimization of Structured SVMs." Proceedings of Machine Learning
        Research. Proceedings of the International Conference on Machine
        Learning (ICML 2016). 2016.
    """

    def __init__(
        self, domain=None, structured_loss=None, dataset_size=1, alpha=0.0001
    ):
        super().__init__(domain=domain)
        if structured_loss is None:
            raise ValueError('Need a structured loss')
        self.structured_loss = structured_loss
        self.dataset_size = dataset_size
        self.alpha = alpha

    @property
    def dual_gap(self):
        if not hasattr(self, 'dual_gap_'):
            raise ValueError('Call fit or partial_fit first')
        return self.dual_gap_

    def _init_w(self, shape):
        return np.zeros(shape, dtype=np.float64)

    def _step(self, w, w_mat, l, l_mat, idx, x, y_true, y_pred):
        if not hasattr(self, 't_'):
            self.t_ = 0
        self.t_ += 1

        phi_y_pred = self.domain.phi(*asarrays(x, y_pred))[0]
        phi_y_true = self.domain.phi(*asarrays(x, y_true))[0]
        psi = phi_y_true - phi_y_pred

        d = psi.shape[0]
        if w is None:
            w = self._init_w(d)
        if w_mat is None:
            w_mat = np.array([])
        if l is None:
            l = 0.0
        if l_mat is None:
            l_mat = np.array([])

        i = hashkey(x, y_true)
        if i not in idx:
            idx[i] = w_mat.shape[0]
            if w_mat.shape[0] > 0:
                w_mat = np.vstack((w_mat, self._init_w(d).reshape(1, -1)))
            else:
                w_mat = self._init_w(d).reshape(1, -1)
            l_mat = np.append(l_mat, 0.0)

        dataset_size = max([self.dataset_size, w_mat.shape[0]])

        ws = ((self.alpha * dataset_size) ** -1) * psi
        ls = self.structured_loss(*asarrays(y_true, y_pred))
        ls /= dataset_size

        w_diff = w_mat[idx[i]] - ws
        dual_gap = self.alpha * (w_diff).dot(w) - l_mat[idx[i]] + ls
        norm_squared = np.power(np.linalg.norm(w_diff), 2)
        if norm_squared == 0:
            gamma = 1.0
        else:
            gamma = dual_gap / (self.alpha * norm_squared)
            gamma = np.max([0.0, np.min([1.0, gamma])])

        w_i = np.copy(w_mat[idx[i]])
        l_i = l_mat[idx[i]]
        w_mat[idx[i]] = (1 - gamma) * w_mat[idx[i]] + gamma * ws
        l_mat[idx[i]] = (1 - gamma) * l_mat[idx[i]] + gamma * ls

        w = w + w_mat[idx[i]] - w_i
        l = l + l_mat[idx[i]] - l_i
        return w, w_mat, l, l_mat, idx, dual_gap

    def partial_fit(self, X, Y, Y_pred=None):
        if not hasattr(self, 'w_'):
            self.w_ = None
        if not hasattr(self, 'w_mat_'):
            self.w_mat_ = None
        if not hasattr(self, 'l_'):
            self.l_ = None
        if not hasattr(self, 'l_mat_'):
            self.l_mat_ = None
        if not hasattr(self, 'idx_'):
            self.idx_ = {}
        if not hasattr(self, 't_'):
            self.iter_ = 0

        self.iter_ += 1

        w = None
        if self.w_ is not None:
            w = np.copy(self.w_)

        w_mat = None
        if self.w_mat_ is not None:
            w_mat = np.copy(self.w_mat_)

        l = None
        if self.l_ is not None:
            l = np.copy(self.l_)

        l_mat = None
        if self.l_mat_ is not None:
            l_mat = np.copy(self.l_mat_)

        idx = None
        if self.idx_ is not None:
            idx = self.idx_

        model = self.model

        infer_times = []
        learn_times = []

        log = get_logger(__name__)

        for i, (x, y_true) in enumerate(zip(X, Y)):

            # Inference
            start = _time()
            if Y_pred is None:
                y_pred = model.predict(
                    *asarrays(x, y_true), problem='loss_augmented_map'
                )
            else:
                y_pred = Y_pred[i]
            infer_time = _time() - start
            infer_times.append(infer_time)

            # Weight update
            start = _time()
            w, w_mat, l, l_mat, idx, dual_gap = self._step(
                w, w_mat, l, l_mat, idx, x, y_true, y_pred
            )
            learn_time = _time() - start
            learn_times.append(learn_time)

            model = LinearModel(self.domain, w)

            log.debug('''\
                Iteration {self.iter_:>2d}, weights update
                x       = {x}
                y_true  = {y_true}
                y_pred  = {y_pred}
                w       = {w}
            ''', locals())

        self.w_ = w
        self.w_mat_ = w_mat
        self.l_ = l
        self.l_mat_ = l_mat
        self.idx_ = idx
        self.dual_gap_ = dual_gap
        self.model_ = LinearModel(self.domain, self.w_)
        return self

    # Alias
    fit = partial_fit

    def loss(self, X, Y_true, Y_pred):
        loss = super().loss(X, Y_true, Y_pred)
        if self.structured_loss is not None:
            loss += self.structured_loss(Y_true, Y_pred)
        return loss

