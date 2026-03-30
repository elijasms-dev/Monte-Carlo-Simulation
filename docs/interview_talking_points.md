# Interview Talking Points

## The 30-second summary

This project is a small derivatives research sandbox built around Monte Carlo methods. It prices vanilla, path-dependent, American, and multi-asset options; validates results against analytical benchmarks where possible; and exposes experiment commands for convergence, scenario analysis, and portfolio-style reporting.

## What to emphasize

- You did not stop at "I can simulate a European call."
- You added variance reduction because practical pricing is about error per unit of compute, not just correctness.
- You separated products, engines, and experiment workflows so the repo looks like research infrastructure rather than a coding exercise.
- You included both analytical checks and Monte Carlo-only products, which shows model selection judgment.
- You compared pseudo-random sampling with quasi-Monte Carlo instead of assuming "Monte Carlo" is one fixed technique.
- You went past flat volatility and added a stochastic-volatility workflow that can both generate a smile and calibrate to a market snapshot.

## Strong discussion angles

### Variance reduction

Explain why antithetic variates are cheap and broadly useful, while control variates become powerful when you have a strongly correlated anchor with known expectation.

### Quasi-Monte Carlo

Explain that low-discrepancy sequences aim to cover the integration domain more evenly than pseudo-random sampling, which can improve convergence behavior even though implementation overhead and dimensionality still matter.

### Greeks

Explain why finite-difference Greeks are noisy and why common random numbers help stabilize the estimates.

### American options

Explain the Longstaff-Schwartz intuition:

- simulate paths
- estimate continuation values by regression
- exercise when intrinsic value dominates continuation

Also mention that under the no-dividend GBM assumption, American calls should match European calls.

### Basket options

Explain why arithmetic baskets generally need Monte Carlo, but geometric baskets give you a closed-form proxy that can double as a control variate.

### Heston stochastic volatility

Explain that constant-volatility Black-Scholes is often too rigid for market smiles and skew, so the Heston workflow introduces a stochastic variance process with correlated spot/variance shocks. Mention that negative correlation is what produces the familiar downward equity skew in the implied-vol surface.

### Market calibration

Explain that building a stochastic-volatility model is only half the story. The more relevant question is whether the model can fit a market surface. Talk through:

- why calibration is easier with a deterministic characteristic-function pricer than with noisy Monte Carlo estimates
- why you report both price-space and implied-volatility errors
- why parameter stability and data quality matter just as much as the optimizer
- what simplifying assumptions the bundled snapshot makes, such as a flat short rate and approximate dividend yield

### Experiment workflows

Point out that the convergence, smile, calibration, surface, and report commands are there to study model behavior systematically, not just print a single price.

## Good follow-up lines if they ask "what next?"

- Sobol sequences or Brownian-bridge constructions on top of the current Halton workflow
- local-volatility calibration on top of the current Heston workflow
- American option basis selection and regularization
- basket options with time-varying covariance inputs
- performance comparisons between Python, `numba`, and C++
