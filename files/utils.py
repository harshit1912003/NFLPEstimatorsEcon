import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

def rho_of_tau(tau):
    """Compute rho(tau) = 2*Phi(tau) - 1."""
    return 2 * norm.cdf(tau) - 1

def lambda_of_tau(tau):
    """Compute lambda(tau) = (tau^2 - 1)*log(tau) - 1."""
    return (tau**2 - 1) * np.log(tau) - 1

def omega_of_tau(tau):
    """Compute omega(tau) based on rho and lambda."""
    rho = rho_of_tau(tau)
    lam = lambda_of_tau(tau)

    return 1.0 / (rho + 2 * norm.pdf(tau) * tau * np.log(tau) / lam)

def tau_of_omega(omega):
    """
    Given omega, find the corresponding tau by solving:
      omega_of_tau(tau) = omega
    for tau in [1.699009, 8]. omega is truncated between 0 and omega_of_tau(8).
    """
    omega = np.clip(omega, 0.0, omega_of_tau(8.0))
    def func(tau):
        return omega_of_tau(tau) - omega
    return brentq(func, 1.699009, 8.0, xtol=1e-12)
