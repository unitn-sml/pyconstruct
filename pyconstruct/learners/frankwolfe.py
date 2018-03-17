
from .base import BaseLearner
from ..utils import get_logger


__all__ = ['BlockCoordinateFrankWolfe']


class BlockCoordinateFrankWolfe(BaseLearner):

    def __init__(
        self, *, domain=None, inference='loss_augmented_map',
        structured_loss=None, hashify=None
    ):
        super().__init__(domain=domain)
        if self.hashify is None:
            raise ValueError('Need an hashify function')
        self.inference = inference
        self.structured_loss = structured_loss
        self.hashify = hashify
        self.n_samples = n_samples

    @property
    def dual_gap(self):
        if not hasattr(self, 'dual_gap_'):
            raise ValueError('Call fit or partial_fit first')
        return self.dual_gap_

    def _init_w(self, shape):
        return np.zeros(shape, dtype=np.float64)

    def _step(self, w, w_mat, l, l_mat, idx, x, y_true, y_pred, n_samples=1):
        if not hasattr(self, 't_'):
            self.t_ = 0
        self.t_ += 1

        phi_y_pred = self.domain.phi(np.array([x]), np.array([y_pred]))[0]
        phi_y_true = self.domain.phi(np.array([x]), np.array([y_true]))[0]
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

        i = self.hashify(x, y)
        if i not in idx:
            idx[i] = w_mat.shape[0]
            w_mat = np.vstack((w_mat, self._init_w(d).reshape(1, -1)))
            l_mat = np.append(l_mat, 0.0)

        n_samples = max([n_samples, w_mat.shape[0]])

        ws = ((self.alpha * n_samples) ** -1) * psi
        ls = self.loss(np.array([x]), np.array([y_true]), np.array([y_pred]))[0]
        ls /= n_samples

        w_diff = w_mat[i] - ws
        dual_gap = self.alpha * (w_diff).dot(w) - l_mat[i] + ls
        gamma = dual_gap / (self.alpha * np.linalg.norm(w_diff) ** 2)
        gamma = np.max([0, np.min([1, gamma])])

        w_i = np.copy(w_mat[i])
        l_i = l_mat[i]
        w_mat[i] = (1 - gamma) * w_mat[i] + gamma * ws
        l_mat[i] = (1 - gamma) * l_mat[i] + gamma * ls

        w = w + w_mat[i] - w_i
        l = l + l_mat[i] - l_i
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

        model = self.model

        infer_times = []
        learn_times = []

        for i, (x, y_true) in enumerate(zip(X, Y)):

            # Inference
            start = _time()
            if Y_pred is None:
                y_pred = model.predict(np.array([x]), problem=self.inference)
            else:
                y_pred = Y_pred[i]
            infer_time = _time() - start
            learn_times.append(learn_time)

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

