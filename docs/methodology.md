# Methodology Notes

## Model

The project assumes a risk-neutral geometric Brownian motion:

```text
dS_t = r S_t dt + sigma S_t dW_t
```

For European payoffs, a one-step exact simulation is sufficient because the terminal distribution is known in closed form. For Asian and barrier payoffs, the engine uses multi-step monitoring because the whole path matters.

## Payoffs

The current implementation supports:

- European call and put
- arithmetic-average Asian call and put
- up-and-out barrier call and put
- American vanilla call and put under the same GBM dynamics
- correlated arithmetic and geometric basket options

That mix is intentional. It shows easy-to-benchmark vanilla pricing, path-dependent cases where simulation is genuinely necessary, early-exercise logic, and a multi-asset setting with correlated factors.

## Variance reduction

### Antithetic variates

The engine pairs each Gaussian draw with its negative. Under GBM this often reduces variance for monotone payoffs at almost no conceptual cost.

### Quasi-Monte Carlo

The toolkit also supports Halton low-discrepancy sampling. The sequence is transformed through an inverse normal CDF so it can drive GBM path generation while preserving deterministic coverage of the unit cube.

This is useful for discussing:

- why lower-discrepancy sampling can reduce integration error
- why randomized shifts matter for practical reproducibility
- why convergence quality and raw runtime are both relevant when evaluating a method

### Control variate

The control variate is the discounted terminal spot:

```text
e^{-rT} S_T
```

Its expectation under the risk-neutral measure is `S_0`, which is known exactly. The engine estimates the optimal coefficient from the simulated sample covariance and variance.

This is especially useful for vanilla payoffs because the payoff is strongly correlated with the terminal spot.

### Geometric basket control variate

For arithmetic basket options, the engine also supports a geometric basket control variate. The geometric basket has a closed-form Black-Scholes-style price under correlated GBM, so it provides a natural anchor for reducing Monte Carlo noise in the arithmetic basket estimator.

## Greeks

Greeks are estimated with finite differences and common random numbers:

- Delta and Gamma from spot bumps
- Vega from volatility bumps

Reusing the same random draws across bumped valuations reduces noise and makes the estimates more interview-worthy than naive bump-and-revalue.

## American exercise

American vanilla options are priced with the Longstaff-Schwartz least-squares Monte Carlo algorithm.

The implementation:

- simulates full GBM paths
- regresses discounted continuation values on polynomial basis functions of spot
- compares continuation estimates against immediate exercise value
- records the exercise premium over the European benchmark

Under the current assumptions, American calls on non-dividend-paying stocks are handled using the no-early-exercise identity.

## Basket options

The basket workflow models correlated terminal asset values using a Cholesky factorization of the correlation matrix.

Two payoff families are included:

- arithmetic basket options, which require Monte Carlo
- geometric basket options, which admit a closed-form benchmark

That pairing is useful in interviews because it shows you understand both the modeling and the validation path.

## Heston stochastic volatility

The Heston workflow models variance as its own stochastic process:

```text
dS_t = r S_t dt + sqrt(v_t) S_t dW_t^S
dv_t = kappa (theta - v_t) dt + xi sqrt(v_t) dW_t^v
```

with correlation `rho` between the spot and variance shocks.

The implementation now supports two complementary valuation paths:

- a full-truncation Euler Monte Carlo scheme for pathwise experimentation
- a semi-closed-form characteristic-function pricer for fast deterministic calibration and smile generation

The Monte Carlo side uses a full-truncation Euler scheme:

- negative variance values are clipped before they enter the drift and diffusion terms
- spot and variance shocks are correlated pathwise
- the resulting prices are translated into Black-Scholes implied vols for easier interpretation

The characteristic-function side matters because calibration is much easier to discuss when the valuation engine is deterministic, fast, and not blurred by Monte Carlo noise.

This is a useful frontier because it lets the project discuss volatility smiles and skew rather than staying trapped in a constant-volatility world.

## Market calibration

The calibration workflow fits Heston parameters to a bundled historical SPY option snapshot.

The current setup:

- loads a cross-section of call quotes across multiple expiries and strikes
- uses bid/ask mid prices as the calibration target
- translates both market and model prices into implied vols for diagnostics
- searches over `v0`, `theta`, `kappa`, `xi`, and `rho`
- scores candidates with a normalized price RMSE plus an implied-volatility penalty

The optimizer is intentionally lightweight and dependency-free:

- an initial global random search explores broad parameter ranges
- shrinking local perturbations refine the best candidate
- a soft Feller-condition penalty discourages obviously poor variance dynamics

This is not meant to replace production calibration infrastructure. It is meant to show the right instincts:

- start from a deterministic pricing engine
- fit against a transparent objective
- report both parameter values and fit quality
- keep the data provenance visible

## Convergence studies

The study workflow compares:

- price
- standard error
- absolute error versus Black-Scholes when available
- runtime
- efficiency gain based on variance-time reduction relative to the naive estimator

That last metric is useful because a variance-reduction method is only really better if it improves accuracy enough to justify its runtime cost.

## Scenario surfaces

The surface workflow prices a grid of spot and volatility combinations and exports the results as CSV.

This is useful for:

- stress testing intuition
- visualizing non-linear sensitivity
- generating charts or tables for a portfolio or interview walkthrough

## Research reports

The report workflow combines convergence, American-option, basket-option, stochastic-volatility, calibration, and surface outputs into a single markdown artifact.

That artifact is intentionally portfolio-friendly:

- it can be generated directly from the code
- it keeps the project reproducible
- it gives you something concrete to show beyond source files alone

The report now also includes Heston smile and market-calibration sections so the generated artifact demonstrates both synthetic model behavior and real-snapshot fitting.
