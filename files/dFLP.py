import numpy as np
from scipy.stats import norm
from files.utils import rho_of_tau, lambda_of_tau, omega_of_tau, tau_of_omega


def dFLP(y, omega, mu=0.0, sigma=1.0):
    """
    Compute the filtered-log-Pareto (FLP) density for observation y.
    If omega == 1, returns zeros.
    """
    if omega == 1:
        return np.zeros_like(np.atleast_1d(y))
    tau = tau_of_omega(omega)
    lam = lambda_of_tau(tau)
    z = np.abs((y - mu) / sigma)

    indic = (z > tau).astype(float)

    z_adjusted = np.where(indic == 0, tau, z)
    density = indic * omega / sigma / (1 - omega) * (
        norm.pdf(tau) * tau / z_adjusted * (np.log(tau) / np.log(z_adjusted))**(lam + 1) - norm.pdf(z_adjusted)
    )
    return density
