from __future__ import annotations

from pathlib import Path

from .american import price_american_option_lsm
from .basket import BASKET_ARITHMETIC, BasketOptionSpec, build_equicorrelation_matrix, price_basket_option_mc
from .calibration import (
    bundled_market_data_path,
    calibrate_heston_parameters,
    format_calibration_table,
    load_market_quotes_csv,
)
from .heston import HestonSpec, format_heston_smile_table, run_heston_smile_cf
from .pricing import CALL, EUROPEAN, PUT, OptionSpec, SimulationConfig
from .study import format_study_table, run_convergence_study
from .surface import format_surface_table, run_sensitivity_surface


def build_research_report(
    destination: str,
    *,
    seed: int = 7,
    study_paths: list[int] | None = None,
    surface_spots: list[float] | None = None,
    surface_vols: list[float] | None = None,
    american_paths: int = 30_000,
    basket_paths: int = 40_000,
    surface_paths: int = 8_000,
    heston_paths: int = 20_000,
    calibration_data: str | None = None,
) -> Path:
    study_paths = [5_000, 20_000, 50_000, 100_000] if study_paths is None else study_paths
    surface_spots = [80.0, 90.0, 100.0, 110.0, 120.0] if surface_spots is None else surface_spots
    surface_vols = [0.15, 0.20, 0.25, 0.30] if surface_vols is None else surface_vols

    european_spec = OptionSpec(
        spot=100.0,
        strike=100.0,
        rate=0.05,
        volatility=0.2,
        maturity=1.0,
        option_type=CALL,
        payoff=EUROPEAN,
    )
    study_results = run_convergence_study(
        spec=european_spec,
        path_counts=study_paths,
        time_steps=1,
        seed=seed,
    )

    american_result = price_american_option_lsm(
        spec=OptionSpec(
            spot=100.0,
            strike=105.0,
            rate=0.05,
            volatility=0.2,
            maturity=1.0,
            option_type=PUT,
            payoff=EUROPEAN,
        ),
        config=SimulationConfig(
            num_paths=american_paths,
            time_steps=50,
            seed=seed,
            antithetic=True,
            control_variate=False,
            sampling="pseudo",
        ),
    )

    basket_result = price_basket_option_mc(
        spec=BasketOptionSpec(
            spots=(100.0, 95.0, 105.0),
            volatilities=(0.20, 0.25, 0.22),
            weights=(0.4, 0.3, 0.3),
            correlation_matrix=build_equicorrelation_matrix(3, 0.35),
            strike=100.0,
            rate=0.04,
            maturity=1.0,
            option_type=CALL,
            payoff=BASKET_ARITHMETIC,
        ),
        config=SimulationConfig(
            num_paths=basket_paths,
            time_steps=1,
            seed=seed,
            antithetic=True,
            control_variate=False,
            sampling="halton",
        ),
        use_geometric_control_variate=True,
    )

    surface_points = run_sensitivity_surface(
        spec=european_spec,
        config=SimulationConfig(
            num_paths=surface_paths,
            time_steps=1,
            seed=seed,
            antithetic=True,
            control_variate=True,
            sampling="halton",
        ),
        spots=surface_spots,
        volatilities=surface_vols,
        include_greeks=True,
    )

    heston_smile = run_heston_smile_cf(
        spec=HestonSpec(
            spot=100.0,
            strike=100.0,
            rate=0.03,
            maturity=1.0,
            initial_variance=0.04,
            long_run_variance=0.04,
            mean_reversion=2.0,
            vol_of_vol=0.55,
            correlation=-0.7,
            option_type=CALL,
        ),
        strikes=[80.0, 90.0, 100.0, 110.0, 120.0],
    )

    calibration_quotes = load_market_quotes_csv(
        calibration_data or bundled_market_data_path()
    )
    calibration_result = calibrate_heston_parameters(
        calibration_quotes,
        seed=seed,
        global_samples=36,
        search_rounds=5,
        local_samples=20,
        integration_points=224,
    )

    report = f"""# Quant Monte Carlo Research Report

This report was generated directly from the repository code. It summarizes the current research workflows and gives a compact artifact you can point to in applications or interviews.

## 1. European Option Convergence Study

The table below compares pseudo-random Monte Carlo, variance-reduced estimators, and quasi-Monte Carlo Halton sampling on a one-step European call benchmark.

```text
{format_study_table(study_results)}
```

## 2. American Put via Longstaff-Schwartz

- Price: `{american_result.price:.5f}`
- 95% CI: `[{american_result.confidence_interval[0]:.5f}, {american_result.confidence_interval[1]:.5f}]`
- European reference: `{american_result.european_reference_price:.5f}`
- Early exercise premium: `{american_result.premium_over_european:.5f}`
- Early exercise ratio: `{100.0 * american_result.early_exercise_ratio:.2f}%`

## 3. Correlated Basket Option

- Product: arithmetic basket call on three assets
- Method: `{basket_result.method}`
- Price: `{basket_result.price:.5f}`
- Std. error: `{basket_result.standard_error:.5f}`
- 95% CI: `[{basket_result.confidence_interval[0]:.5f}, {basket_result.confidence_interval[1]:.5f}]`

## 4. Spot/Volatility Surface

```text
{format_surface_table(surface_points)}
```

## 5. Heston Volatility Smile

```text
{format_heston_smile_table(heston_smile)}
```

## 6. Heston Calibration to a Market Snapshot

- Dataset: bundled SPY call snapshot from 2026-03-11 across five maturities and five strikes
- Quotes fitted: `{calibration_result.num_quotes}`
- RMSE price: `{calibration_result.rmse_price:.5f}`
- RMSE implied vol: `{"-" if calibration_result.rmse_implied_volatility is None else f"{calibration_result.rmse_implied_volatility:.5f}"}`
- Fitted parameters: `v0={calibration_result.parameters.initial_variance:.4f}`, `theta={calibration_result.parameters.long_run_variance:.4f}`, `kappa={calibration_result.parameters.mean_reversion:.4f}`, `xi={calibration_result.parameters.vol_of_vol:.4f}`, `rho={calibration_result.parameters.correlation:.4f}`

```text
{format_calibration_table(calibration_result.fitted_points, max_rows=10)}
```

## 7. Takeaways

- variance reduction improves error per unit of compute, especially when paired with strong analytical anchors
- early exercise matters for puts and can be studied directly through regression-based continuation estimates
- correlated multi-asset structures create natural use cases for basket benchmarks and control variates
- stochastic volatility generates strike-dependent implied vols, which makes the toolkit meaningfully closer to market reality than flat-vol GBM
- calibrating Heston to a market snapshot is a stronger research signal than generating a synthetic smile in isolation
- scenario surfaces provide a clean way to discuss non-linearity, sensitivities, and stress behavior
"""

    output_path = Path(destination)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    return output_path
