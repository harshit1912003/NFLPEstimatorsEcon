import numpy as np
from scipy.stats import norm
from files.dNFLP import dNFLP
from files.thresh import threshold

def optim_NFLP_reg(y, X, intercept=True, omega=0.80, beta=None, sigma=None, tol=1e-9, max_iter=500):
    y = np.asarray(y).flatten()
    n = len(y)
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if intercept:
        X = np.hstack([np.ones((n, 1)), X])

    p = X.shape[1]

    if beta is None or sigma is None:
        beta_ols, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
        if beta is None:
            beta = beta_ols
        if sigma is None:
            sigma = np.std(y - X.dot(beta))

    err = 1e10
    iter_count = 0
    omega0, beta0, sigma0 = omega, beta.copy(), sigma
    mu = X.dot(beta)

    while err > tol and iter_count < max_iter:
        iter_count += 1
        if omega == 1:
            Pi = np.ones(n)
        else:
            Pi = omega * norm.pdf(y, mu, sigma) / dNFLP(y, omega, mu, sigma)
            Pi = np.minimum(Pi, 1.0)
            omega = np.mean(Pi)
            if omega <= 0.5:
                Pi = np.ones(n)
                omega = 1.0

        W = np.diag(Pi)
        XtW = X.T * Pi
        beta = np.linalg.solve(XtW.dot(X), XtW.dot(y))
        mu = X.dot(beta)
        sigma = np.sqrt(np.sum(Pi * (y - mu)**2) / (omega * n - p))
        sigma = max(sigma, 1e-9)
        err = max(abs(omega0 - omega), np.max(np.abs(beta0 - beta)), abs(sigma0 - sigma))
        omega0, beta0, sigma0 = omega, beta.copy(), sigma

    return np.concatenate(([omega], beta, [sigma]))

def lm_NFLP(y, X, intercept=True, nb_start=10, omega=0.80, beta=None, sigma=None):
    base_solution = optim_NFLP_reg(y, X, intercept, omega, beta, sigma)
    solutions = [base_solution]
    n, p = np.shape(X)[0], X.shape[1] + (1 if intercept else 0)

    for i in range(1, nb_start):
        # print(i)
        if beta is None:
            beta_ols, residuals, rank, s = np.linalg.lstsq(np.hstack([np.ones((len(y), 1)), X]) if intercept else X, y, rcond=None)
            beta_start = beta_ols + np.random.normal(scale=s, size=beta_ols.shape)
        else:
            beta_start = beta + np.random.normal(scale=sigma, size=len(beta))
        sol = optim_NFLP_reg(y, X, intercept, omega, beta_start, sigma)
        solutions.append(sol)

    sols = np.array(solutions)
    valid = sols[:,0] > 0.5
    if np.any(valid):
        sols_valid = sols[valid]
        best_index = np.argmin(sols_valid[:,-1])
        best_solution = sols_valid[best_index]
    else:
        best_solution = sols[0]

    omega_hat = best_solution[0]
    beta_hat = best_solution[1:-1]
    sigma_hat = best_solution[-1]

    X_design = np.hstack([np.ones((len(y), 1)), X]) if intercept else X
    y_pred = X_design.dot(beta_hat)
    residuals = y - y_pred
    std_resid = residuals / sigma_hat

    Pi = omega_hat * norm.pdf(y, y_pred, sigma_hat) / dNFLP(y, omega_hat, y_pred, sigma_hat)
    Pi = np.minimum(Pi, 1.0)

    return {
        'omega': omega_hat,
        'beta': beta_hat,
        'sigma': sigma_hat,
        'fitted': y_pred,
        'residuals': residuals,
        'std_resid': std_resid,
        'Pi': Pi
    }
