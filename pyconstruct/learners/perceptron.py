
import numpy as np

from ..utils import asarrays
from .subgradient import SSG


__all__ = ['StructuredPerceptron']


class StructuredPerceptron(SSG):
    """A simple structured perceptron algorithm from [1]_.

    Parameters
    ----------
    domain : BaseDomain
        The domain of the data.

    References
    ----------
    .. [1] Collins, Michael. "Discriminative training methods for hidden markov
        models: Theory and experiments with perceptron algorithms." EMNLP
        (2002).
    """
    def __init__(self, domain=None, **kwargs):
        super().__init__(
            domain, projection=None, alpha=0.0, learning_rate='constant',
            inference='map', **kwargs
        )

    def _step(self, x, y_true, y_pred, phi_y_true, phi_y_pred, w, eta):
        score_true = self.model.decision_function(*asarrays(x, y_true))
        score_pred = self.model.decision_function(*asarrays(x, y_pred))
        if w is None or score_true <= score_pred:
            return super()._step(
                x, y_true, y_pred, phi_y_true, phi_y_pred, w, eta
            )
        return 0.0

