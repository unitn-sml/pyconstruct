
import numpy as np


__all__ = ['l2', 'l1']


def l2(w, radius=1.0):
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


def l1(v, radius=1.0):
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
