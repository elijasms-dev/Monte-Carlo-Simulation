"""Quant-focused Monte Carlo pricing, basket, and exercise-analysis toolkit."""

from .american import AmericanOptionResult, price_american_option_lsm
from .basket import (
    BASKET_ARITHMETIC,
    BASKET_GEOMETRIC,
    BasketOptionSpec,
    build_equicorrelation_matrix,
    geometric_basket_price,
    price_basket_option_mc,
)
from .pricing import (
    ASIAN_ARITHMETIC,
    CALL,
    EUROPEAN,
    PUT,
    UP_AND_OUT,
    GreeksEstimate,
    MonteCarloResult,
    OptionSpec,
    SimulationConfig,
    StudyResult,
    black_scholes_delta,
    black_scholes_gamma,
    black_scholes_price,
    black_scholes_vega,
    build_method_config,
    price_european_option_mc,
    price_option_mc,
)
from .report import build_research_report
from .sampling import halton_uniforms, inverse_normal_cdf
from .study import DEFAULT_METHODS, format_study_table, run_convergence_study, write_study_csv
from .surface import SurfacePoint, format_surface_table, run_sensitivity_surface, write_surface_csv

__all__ = [
    "ASIAN_ARITHMETIC",
    "AmericanOptionResult",
    "BASKET_ARITHMETIC",
    "BASKET_GEOMETRIC",
    "BasketOptionSpec",
    "CALL",
    "DEFAULT_METHODS",
    "EUROPEAN",
    "GreeksEstimate",
    "MonteCarloResult",
    "OptionSpec",
    "PUT",
    "SimulationConfig",
    "StudyResult",
    "SurfacePoint",
    "UP_AND_OUT",
    "black_scholes_delta",
    "black_scholes_gamma",
    "black_scholes_price",
    "black_scholes_vega",
    "build_equicorrelation_matrix",
    "build_method_config",
    "build_research_report",
    "format_study_table",
    "format_surface_table",
    "geometric_basket_price",
    "halton_uniforms",
    "inverse_normal_cdf",
    "price_american_option_lsm",
    "price_basket_option_mc",
    "price_european_option_mc",
    "price_option_mc",
    "run_convergence_study",
    "run_sensitivity_surface",
    "write_study_csv",
    "write_surface_csv",
]
