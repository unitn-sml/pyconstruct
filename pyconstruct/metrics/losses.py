
from ..utils import broadcast
from sklearn.metrics import hamming_loss


__all__ = ['hamming']


def _hamming(y_true, y_pred, key=None):
    if key:
        y_true = y_true[key]
        y_pred = y_pred[key]
    loss = hamming_loss(y_true, y_pred)
    return loss


def hamming(Y_true, Y_pred, key=None, n_jobs=1):
    """Element-wise Hamming distance.

    Parameters
    ----------
    Y_true : array-like or [array-like]
        The true output objects.
    Y_pred : array-like or [array-like]
        The predicted objects.
    key : str
        The key of the objects which to compute the hamming distance on.
    n_jobs : int
        The number of parallel thread to use to compute the losses.
    Returns
    -------
    hamming : float or [float]
        The hamming distance(s)
    """
    return broadcast(_hamming, Y_true, Y_pred, key=key, n_jobs=n_jobs)

