import numpy as np
from scipy.stats import norm
from files.utils import rho_of_tau, lambda_of_tau, omega_of_tau, tau_of_omega

def dNFLP(y, omega, mu=0.0, sigma=1.0, log=False):
    """
    Compute the density (or log density) of the N-FLP distribution.
    For omega == 1, this is simply the normal density.
    """
    if omega == 1:
        return norm.logpdf(y, mu, sigma) if log else norm.pdf(y, mu, sigma)
    tau = tau_of_omega(omega)
    lam = lambda_of_tau(tau)
    z = np.abs((y - mu) / sigma)

    indic_C = (z <= tau).astype(float)
    log_density_C = indic_C * norm.logpdf(z)

    z_adjusted = np.where(indic_C == 1, tau, z)
    log_density_T = (1 - indic_C) * (norm.logpdf(tau) + np.log(tau) - np.log(z_adjusted) + (lam + 1) * (np.log(np.log(tau)) - np.log(np.log(z_adjusted))))
    log_density = np.log(omega) - np.log(sigma) + log_density_C + log_density_T
    return np.exp(log_density) if not log else log_density
