# Monte Carlo Simulation

A small Python project for building and discussing Monte Carlo pricing workflows in quant internship interviews.

## Why this repo

This starter focuses on the kind of material that comes up in quant screens and internship take-homes:

- simulating geometric Brownian motion
- pricing European options with Monte Carlo
- comparing simulation output against a Black-Scholes benchmark
- reporting standard errors and confidence intervals

## Project layout

```text
src/monte_carlo_simulation/
  pricing.py      # Monte Carlo pricer and Black-Scholes benchmark
  cli.py          # Simple command-line entrypoint
tests/
  test_pricing.py # Sanity checks for the pricing logic
```

## Quick start

1. Create a virtual environment.
2. Install the package in editable mode.
3. Run the CLI or the tests.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e .
python -m monte_carlo_simulation --paths 100000 --seed 7
python -m unittest discover -s tests
```

## Example output

```text
Monte Carlo price : 10.4732
Std. error        : 0.0463
95% CI            : [10.3824, 10.5640]
Black-Scholes     : 10.4506
Absolute error    : 0.0226
```

## Good follow-on extensions

- add Asian, barrier, or basket option payoffs
- support quasi-random sampling or control variates
- compare convergence speed across variance reduction techniques
- add notebook-based visualizations of simulated paths
