# Monte Carlo Simulation for Quant Interviews

This repository has grown into a compact quant research toolkit rather than a single pricing script. It now covers vanilla and path-dependent single-asset options, correlated basket options, American early-exercise logic via Longstaff-Schwartz, and experiment workflows for convergence and scenario surfaces.

## What makes it stand out

- vectorized Monte Carlo engine built with `numpy`
- pseudo-random and quasi-Monte Carlo Halton sampling
- path-dependent pricing for arithmetic Asian and up-and-out barrier options
- variance reduction with antithetic variates, terminal-spot control variates, and geometric-basket control variates
- finite-difference Greeks using common random numbers
- Longstaff-Schwartz implementation for American vanilla options
- correlated basket option pricing with a closed-form geometric basket benchmark
- convergence-study, surface-generation, and report-generation workflows that export clean artifacts
- regression tests that check pricing accuracy, variance reduction, early-exercise logic, and multi-asset behavior

## Project structure

```text
src/monte_carlo_simulation/
  pricing.py    # single-asset Monte Carlo engine and Black-Scholes analytics
  sampling.py   # pseudo-random and Halton quasi-random sampling tools
  american.py   # Longstaff-Schwartz American option pricing
  basket.py     # correlated basket option pricing and geometric benchmark
  study.py      # convergence-study workflow and CSV export
  surface.py    # scenario surface generation across spot/volatility grids
  report.py     # markdown research report generator
  cli.py        # command-line interface across all workflows
tests/
  test_pricing.py
docs/
  methodology.md
  interview_talking_points.md
```

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e .
python -m unittest discover -s tests
```

## Example commands

Price a European call with variance reduction and Greek estimates:

```bash
python -m monte_carlo_simulation price --steps 1 --paths 100000 --control-variate
```

Use Halton quasi-Monte Carlo sampling on the same problem:

```bash
python -m monte_carlo_simulation price --steps 1 --paths 100000 --sampling halton --control-variate
```

Price an arithmetic Asian option:

```bash
python -m monte_carlo_simulation price --payoff asian-arithmetic --steps 252 --paths 150000
```

Price an up-and-out barrier call:

```bash
python -m monte_carlo_simulation price --payoff up-and-out --barrier 125 --steps 252 --paths 150000
```

Run a convergence study and export the results:

```bash
python -m monte_carlo_simulation study --steps 1 --path-counts 5000 20000 50000 100000 --csv outputs/european_study.csv
```

Price an American put with Longstaff-Schwartz:

```bash
python -m monte_carlo_simulation american --option-type put --spot 100 --strike 105 --vol 0.2 --steps 50 --paths 30000
```

Price a three-asset arithmetic basket option:

```bash
python -m monte_carlo_simulation basket --spots 100 95 105 --vols 0.20 0.25 0.22 --weights 0.4 0.3 0.3 --paths 50000
```

Generate a spot/volatility surface:

```bash
python -m monte_carlo_simulation surface --steps 1 --paths 8000 --spot-grid 80 90 100 110 120 --vol-grid 0.15 0.20 0.25 0.30 --csv outputs/surface.csv
```

Generate a markdown research report:

```bash
python -m monte_carlo_simulation report --output outputs/research_report.md
```

## Why a quant team should care

- the repo includes both analytical validation and simulation-heavy products, which shows judgment about when Monte Carlo is necessary
- the toolkit compares pseudo-random Monte Carlo against quasi-Monte Carlo instead of treating all sampling the same
- the American module demonstrates awareness of early exercise and regression-based continuation estimation
- the basket module shows comfort with correlated risk factors and multi-asset payoff construction
- the experiment commands make it easy to talk about runtime-versus-accuracy tradeoffs like a researcher, not just a coder
- the report workflow produces a polished artifact that can support a portfolio, application, or interview walkthrough
- the codebase is organized as a reusable library with experiment entrypoints rather than disconnected scripts

## Good interview discussion threads

- why a one-step GBM simulation is exact for European options but not for barrier monitoring or Asian averaging
- when quasi-Monte Carlo improves convergence in practice and why lower discrepancy can still trade off against implementation overhead
- why control variates work well when the anchor payoff is strongly correlated with the target payoff
- how common random numbers reduce noise in finite-difference Greek estimation
- why American calls on non-dividend-paying stocks should not be exercised early, but puts often should
- why a geometric basket admits a closed-form proxy while an arithmetic basket still needs simulation

## Suggested next frontier

- Sobol sequences and Brownian-bridge ordering on top of the current Halton workflow
- Heston or local-volatility path generation
- Longstaff-Schwartz with richer basis functions or regularization
- basket options with time-dependent correlations
- a notebook or lightweight dashboard that plots convergence and surface outputs directly
