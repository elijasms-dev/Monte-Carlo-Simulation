# Market Data Notes

## Bundled snapshot

The calibration demo uses a bundled SPY call snapshot dated `2026-03-11`.

The current CSV contains:

- strikes `673` through `677`
- expiries `2026-03-16`, `2026-03-27`, `2026-04-17`, `2026-04-30`, and `2026-05-15`
- spot level `675.77`
- bid/ask quotes converted to mid prices inside the loader

## Source trail

The option quotes were transcribed from the public SPY option-chain snapshot published on Optionistics:

- `https://www.optionistics.com/quotes/stock-option-chains/SPY`

The risk-free rate was approximated with the short end of the U.S. Treasury daily par yield curve for the same date:

- `https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve`

## Modeling simplifications

This dataset is intentionally lightweight and portfolio-friendly, not institutional-grade.

The main simplifications are:

- a flat continuously compounded rate of `3.73%` across the fitted maturities
- a flat dividend yield assumption of `1.50%`
- call-only calibration rather than a full call-and-put surface
- no cleaning for stale quotes, crossed markets, or liquidity screens beyond using tight near-the-money strikes

Those choices are acceptable for a portfolio project because they keep the workflow reproducible while still showing the important research pattern:

1. gather a transparent market snapshot
2. define a calibration objective
3. fit a stochastic-volatility model
4. report error metrics and discuss limitations honestly
