import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from files.dNFLP import dNFLP


def threshold(omega):
    """
    Compute the outlier threshold. For omega very close to 1, return infinity.
    Otherwise, solve for rr in: omega*dnorm(rr)/dNFLP(rr) = 0.5
    over the interval [2.466466, 8].
    """
    if omega > 1 - 1e-12:
        return np.inf

    omega = np.clip(omega, 1e-12, 1 - 1e-12)
    def g(rr):
        return omega * norm.pdf(rr, 0.0, 1.0) / dNFLP(rr, omega, 0.0, 1.0) - 0.5
    return brentq(g, 2.466466, 8.0, xtol=1e-12)



