# N-FLP Robust Linear Regression

## Project Overview

This repository presents a Python implementation of the **N-FLP (Normal - Filtered Log-Pareto) mixture model for robust linear regression**, as proposed by Alain Desgagné in the paper *"Efficient and Robust Estimation of Linear Regression with Normal Errors"* (2019, arXiv:1909.07719v1).

The project aims to provide a practical toolkit for estimating linear regression parameters that are resilient to outliers, offering an improvement in efficiency over traditional robust methods while maintaining performance comparable to Ordinary Least Squares (OLS) in the absence of contamination.

## Motivation

Linear regression with normally distributed errors, $y_i = \mathbf{x}_i^T \boldsymbol{\beta} + \varepsilon_i$ where $\varepsilon_i \sim \mathcal{N}(0, \sigma^2)$, is a cornerstone of statistical modeling. While OLS estimators for $\boldsymbol{\beta}$ possess desirable properties like minimum variance among unbiased estimators (Gauss-Markov theorem), their performance degrades significantly in the presence of outliers—observations with errors that deviate markedly from the assumed normal distribution.

Existing robust estimators (M, S, LMS, LTS, MM, REWLS) address this sensitivity, but often at the cost of efficiency or with complexities in scale parameter estimation. Desgagné's N-FLP approach offers a compelling alternative by:
1.  Modeling errors as a mixture of a normal distribution and a novel Filtered Log-Pareto (FLP) distribution designed to capture outliers.
2.  Employing an adapted Expectation-Maximization (EM) algorithm for transparent and interpretable parameter estimation.
3.  Providing robust inference capabilities, including outlier identification, confidence intervals, and hypothesis testing.

This implementation seeks to replicate and explore the N-FLP methodology, demonstrating its efficacy on both simulated and real-world economic data.

## The N-FLP Model: Core Concepts

The fundamental idea is to replace the assumption of purely normal errors with a mixture distribution for the error term $\varepsilon_i$:

$$ \varepsilon_i \sim \text{N-FLP}(\omega, 0, \sigma) = \omega \mathcal{N}(0, \sigma^2) + (1 - \omega) \text{FLP}(\omega, 0, \sigma) $$

Where:
-   $\omega \in (0, 1]$ is the mixture weight, representing the proportion of observations arising from the normal component. If $\omega = 1$, the model reduces to the standard normal error model.
-   $\mathcal{N}(0, \sigma^2)$ is the normal distribution with mean 0 and variance $\sigma^2$.
-   $\text{FLP}(\omega, 0, \sigma)$ is the Filtered Log-Pareto distribution, designed to model outliers. Its density is defined for $|z| > \tau$, where $z = \varepsilon_i / \sigma$ is the standardized residual.

The N-FLP density is effectively a normal density down-weighted in its central part ($|z| \le \tau$) and, correspondingly, has its tails thickened to accommodate extreme values. The threshold $\tau > 1.69901$ (defining the outlier region) and the tail's decay rate $\lambda > 0$ are automatically determined as functions of $\omega$.

![nflp](/distribution.png)

### Parameter Estimation
The N-FLP estimators for the regression coefficients $\boldsymbol{\hat{\beta}}$, scale $\hat{\sigma}$, and mixture weight $\hat{\omega}$ are derived using an adapted Expectation-Maximization (EM) algorithm. A key output is the probability $\hat{\pi}_i$ for each observation $y_i$ of belonging to the normal component, given by:

$$
\hat{\pi}\_i = \pi_{\hat{\omega}}(r_i) = \frac{\hat{\omega} \, f_{\mathcal{N}}(y_i \mid \mathbf{x}\_i^\top \hat{\boldsymbol{\beta}}, \hat{\sigma})}{f_{\text{N-FLP}}(y_i \mid \hat{\omega}, \mathbf{x}_i^\top \hat{\boldsymbol{\beta}}, \hat{\sigma})}
$$

where $r_i = (y_i - \mathbf{x}_i^T \boldsymbol{\hat{\beta}}) / \hat{\sigma}$ is the standardized residual.
-   If $|r_i| \le \hat{\tau}$, then $\hat{\pi}_i = 1$.
-   Observations with $\hat{\pi}_i < 0.5$ can be flagged as outliers.

The iterative estimation process alternates between computing $\hat{\omega}, \boldsymbol{\hat{\beta}}, \hat{\sigma}$ and updating the diagonal matrix $\mathbf{D}_{\boldsymbol{\pi}} = \text{diag}(\hat{\pi}_1, \dots, \hat{\pi}_n)$. The estimators are:
-   $\hat{\omega} = \frac{1}{n} \sum_{i=1}^n \hat{\pi}_i$
-   $\boldsymbol{\hat{\beta}} = (\mathbf{X}^T \mathbf{D}\_{\boldsymbol{\pi}} \mathbf{X})^{-1} \mathbf{X}^T \mathbf{D}_{\boldsymbol{\pi}} \mathbf{y}$
-   $\hat{\sigma}^2 = \frac{1}{\sum \hat{\pi}\_i - p} \sum_{i=1}^n \hat{\pi}_i (y_i - \mathbf{x}_i^T \boldsymbol{\hat{\beta}})^2$
    (where $p$ is the number of parameters in $\boldsymbol{\beta}$)

Convergence is achieved when the difference between estimators of two successive iterations is below a tolerance (e.g., $10^{-9}$).

## Implementation Details

*   **Project Structure:**
      *   `files/`
          *   `algo.py`: Contains the main `lm_NFLP` estimation function and the `optim_NFLP_reg` EM-like optimization routine.
          *   `dNFLP.py`: Density function for the N-FLP distribution.
          *   `dFLP.py`: Density function for the FLP distribution.
          *   `thresh.py`: Function to compute outlier thresholds based on $\omega$.
          *   `utils.py`: Helper functions for $\rho(\tau)$, $\lambda(\tau)$, $\omega(\tau)$, and $\tau(\omega)$.
      *   `example.ipynb`: Jupyter notebook demonstrating the N-FLP algorithm on simulated data, comparing it with OLS, and illustrating its robustness to outliers.
      *   `econ.ipynb`: Jupyter notebook applying the N-FLP model to a real-world economics dataset (`sampled_data613.csv`) to analyze wage determinants and identify outliers.
      *   `econpaper.pdf`: The original research paper by Alain Desgagné.
      *   `presentation.pdf`: Our presentation for the course.

## Usage and Examples

### 1. Simulated Data Example (`example.ipynb`)
This notebook:
*   Generates synthetic data from a known linear model with normal errors: $y = 3 + 0.5X + \mathcal{N}(0, 1)$.
*   Fits an OLS model.
*   Fits the N-FLP model using `lm_NFLP`.
*   Compares the true line, OLS fit, and N-FLP fit graphically.
*   Introduces outliers to the data and re-fits both OLS and N-FLP models to demonstrate the robustness of N-FLP.

![compare](/comparision.png)

### 2. Economic Data Application (`econ.ipynb`)
This notebook utilizes the `sampled_data613.csv` dataset (from IPUMS USA 2019 focusing on wage, education, and experience). It performs:
*  A Mincer‑type wage regression:
$$
\log(\mathrm{INCWAGE}) 
  = \beta_0 
  + \beta_1 \mathrm{Years\_Education} 
  + \beta_2 \mathrm{Exp} 
  + \beta_3 \mathrm{Exp}^2 
  + \varepsilon
$$
*   OLS estimation of the model.
*   N-FLP estimation of the model.
*   Analysis of the $\hat{\pi}_i$ values to identify potential outliers.
*   Comparison of Mean Squared Error (SSE) for OLS and N-FLP under varying outlier contamination scenarios (by progressively defining outliers based on $\hat{\pi}_i$ thresholds and by varying outlier percentages).


## Key Features of N-FLP (and this implementation)

*   **Robustness:** Less sensitive to outliers than OLS.
*   **Efficiency:** High efficiency in the absence of outliers (approaching OLS) and improved efficiency over many robust methods in the presence of outliers.
*   **Transparency:** The EM-like algorithm provides interpretable parameters, including $\hat{\omega}$ (estimated proportion of "good" data) and $\hat{\pi}_i$ (probability of $i$-th observation being "good").
*   **Automatic Outlier Region:** The $\tau$ parameter, defining the boundary between normal and outlier regions, is endogenously determined by $\hat{\omega}$.
*   **Scale Estimation:** Provides a robust estimate of the scale parameter $\sigma$.

## Reference

Desgagné, A. (2019). *Efficient and Robust Estimation of Linear Regression with Normal Errors*. arXiv:1909.07719v1 [stat.ME].

## License

MIT 
