# Interview Talking Points

## The 30-second summary

This project is a small derivatives research sandbox built around Monte Carlo methods. It prices vanilla, path-dependent, American, and multi-asset options; validates results against analytical benchmarks where possible; and exposes experiment commands for convergence, scenario analysis, and portfolio-style reporting.

## What to emphasize

- You did not stop at "I can simulate a European call."
- You added variance reduction because practical pricing is about error per unit of compute, not just correctness.
- You separated products, engines, and experiment workflows so the repo looks like research infrastructure rather than a coding exercise.
- You included both analytical checks and Monte Carlo-only products, which shows model selection judgment.
- You compared pseudo-random sampling with quasi-Monte Carlo instead of assuming "Monte Carlo" is one fixed technique.

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

### Experiment workflows

Point out that the convergence, surface, and report commands are there to study model behavior systematically, not just print a single price.

## Good follow-up lines if they ask "what next?"

- Sobol sequences or Brownian-bridge constructions on top of the current Halton workflow
- stochastic-volatility models such as Heston
- American option basis selection and regularization
- basket options with time-varying covariance inputs
- performance comparisons between Python, `numba`, and C++
