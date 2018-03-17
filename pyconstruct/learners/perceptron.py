
import numpy as np

from .subgradient import SSG


__all__ = ['StructuredPerceptron']


class StructuredPerceptron(SSG):
    """A simple structured perceptron algorithm.

    References
    ----------
    .. [collins2002discriminative] Collins, Michael. "Discriminative training
        methods for hidden markov models: Theory and experiments with perceptron
        algorithms." EMNLP (2002).
    """
    def __init__(self, *, domain=None):
        super().__init__(
            domain=domain, projection=None, alpha=0.0, learning_rate='constant',
            inference='map'
        )

    def _step(self, w, x, y_true, y_pred):
        if y_true != y_pred:
            return super()._step(w, x, y_true, y_pred)
        return w

