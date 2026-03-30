from __future__ import annotations

from dataclasses import dataclass
import math
from time import perf_counter

import numpy as np
from numpy.typing import NDArray

from .pricing import CALL, MonteCarloResult, SimulationConfig, black_scholes_price, _validate_config
from .sampling import halton_uniforms, inverse_normal_cdf

BASKET_ARITHMETIC = "basket-arithmetic"
BASKET_GEOMETRIC = "basket-geometric"
SUPPORTED_BASKET_PAYOFFS = (BASKET_ARITHMETIC, BASKET_GEOMETRIC)


@dataclass(frozen=True)
class BasketOptionSpec:
    spots: tuple[float, ...]
    volatilities: tuple[float, ...]
    weights: tuple[float, ...]
    correlation_matrix: tuple[tuple[float, ...], ...]
    strike: float
    rate: float
    maturity: float
    option_type: str = CALL
    payoff: str = BASKET_ARITHMETIC

    @property
    def dimension(self) -> int:
        return len(self.spots)


def build_equicorrelation_matrix(dimension: int, rho: float) -> tuple[tuple[float, ...], ...]:
    if dimension < 1:
        raise ValueError("dimension must be at least 1.")
    if rho <= -1.0 / max(dimension - 1, 1) or rho >= 1.0:
        raise ValueError("rho must produce a positive-definite equicorrelation matrix.")
    matrix = []
    for row in range(dimension):
        matrix.append(tuple(1.0 if row == col else rho for col in range(dimension)))
    return tuple(matrix)


def _validate_basket_spec(spec: BasketOptionSpec) -> None:
    if spec.dimension < 1:
        raise ValueError("basket must contain at least one asset.")
    if len(spec.volatilities) != spec.dimension:
        raise ValueError("volatilities must match the number of spots.")
    if len(spec.weights) != spec.dimension:
        raise ValueError("weights must match the number of spots.")
    if len(spec.correlation_matrix) != spec.dimension:
        raise ValueError("correlation_matrix must have one row per asset.")
    if any(len(row) != spec.dimension for row in spec.correlation_matrix):
        raise ValueError("correlation_matrix must be square.")
    if any(spot <= 0 for spot in spec.spots):
        raise ValueError("all spots must be positive.")
    if any(vol < 0 for vol in spec.volatilities):
        raise ValueError("volatilities cannot be negative.")
    if spec.strike <= 0:
        raise ValueError("strike must be positive.")
    if spec.maturity < 0:
        raise ValueError("maturity cannot be negative.")
    if not math.isclose(sum(spec.weights), 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("weights must sum to 1.0.")
    if spec.payoff not in SUPPORTED_BASKET_PAYOFFS:
        raise ValueError(f"payoff must be one of {SUPPORTED_BASKET_PAYOFFS}.")

    correlation = np.asarray(spec.correlation_matrix, dtype=np.float64)
    if not np.allclose(correlation, correlation.T, atol=1e-10):
        raise ValueError("correlation_matrix must be symmetric.")
    if not np.allclose(np.diag(correlation), 1.0, atol=1e-10):
        raise ValueError("correlation_matrix must have ones on the diagonal.")
    np.linalg.cholesky(correlation)


def _generate_correlated_normals(
    spec: BasketOptionSpec,
    config: SimulationConfig,
) -> NDArray[np.float64]:
    base_paths = math.ceil(config.num_paths / 2) if config.antithetic else config.num_paths
    if config.sampling == "halton":
        uniforms = halton_uniforms(base_paths, spec.dimension, seed=config.seed)
        if config.antithetic:
            uniforms = np.vstack((uniforms, 1.0 - uniforms))[: config.num_paths]
        normals = inverse_normal_cdf(uniforms)
    else:
        rng = np.random.default_rng(config.seed)
        normals = rng.standard_normal((base_paths, spec.dimension), dtype=np.float64)
        if config.antithetic:
            normals = np.vstack((normals, -normals))[: config.num_paths]
    correlation = np.asarray(spec.correlation_matrix, dtype=np.float64)
    cholesky = np.linalg.cholesky(correlation)
    return normals @ cholesky.T


def _basket_payoff(values: NDArray[np.float64], strike: float, option_type: str) -> NDArray[np.float64]:
    if option_type == CALL:
        return np.maximum(values - strike, 0.0)
    return np.maximum(strike - values, 0.0)


def geometric_basket_price(spec: BasketOptionSpec) -> float:
    _validate_basket_spec(spec)
    weighted_log_spot = sum(weight * math.log(spot) for weight, spot in zip(spec.weights, spec.spots))
    variance_weights = np.asarray(spec.weights, dtype=np.float64)
    vol_vector = np.asarray(spec.volatilities, dtype=np.float64)
    correlation = np.asarray(spec.correlation_matrix, dtype=np.float64)
    covariance = np.outer(vol_vector, vol_vector) * correlation
    effective_variance = float(variance_weights @ covariance @ variance_weights)
    weighted_vol_square = float(sum(weight * vol * vol for weight, vol in zip(spec.weights, spec.volatilities)))
    effective_spot = math.exp(
        weighted_log_spot + 0.5 * (effective_variance - weighted_vol_square) * spec.maturity
    )
    effective_volatility = math.sqrt(max(effective_variance, 0.0))
    return black_scholes_price(
        spot=effective_spot,
        strike=spec.strike,
        rate=spec.rate,
        volatility=effective_volatility,
        maturity=spec.maturity,
        option_type=spec.option_type,
    )


def price_basket_option_mc(
    spec: BasketOptionSpec,
    config: SimulationConfig = SimulationConfig(),
    *,
    use_geometric_control_variate: bool = True,
) -> MonteCarloResult:
    _validate_basket_spec(spec)
    _validate_config(config)

    start = perf_counter()
    normals = _generate_correlated_normals(spec, config)
    spots = np.asarray(spec.spots, dtype=np.float64)
    volatilities = np.asarray(spec.volatilities, dtype=np.float64)
    weights = np.asarray(spec.weights, dtype=np.float64)
    drift = (spec.rate - 0.5 * volatilities * volatilities) * spec.maturity
    diffusion = volatilities * math.sqrt(spec.maturity)
    terminal = spots * np.exp(drift + diffusion * normals)
    arithmetic_basket = terminal @ weights
    geometric_basket = np.exp(np.sum(weights * np.log(terminal), axis=1))
    discount = math.exp(-spec.rate * spec.maturity)

    if spec.payoff == BASKET_ARITHMETIC:
        raw_payoffs = discount * _basket_payoff(arithmetic_basket, spec.strike, spec.option_type)
        geometric_payoffs = discount * _basket_payoff(geometric_basket, spec.strike, spec.option_type)
        if use_geometric_control_variate:
            benchmark = geometric_basket_price(
                BasketOptionSpec(
                    spots=spec.spots,
                    volatilities=spec.volatilities,
                    weights=spec.weights,
                    correlation_matrix=spec.correlation_matrix,
                    strike=spec.strike,
                    rate=spec.rate,
                    maturity=spec.maturity,
                    option_type=spec.option_type,
                    payoff=BASKET_GEOMETRIC,
                )
            )
            variance = float(np.var(geometric_payoffs, ddof=1))
            if variance > 0:
                covariance = float(np.cov(raw_payoffs, geometric_payoffs, ddof=1)[0, 1])
                beta = covariance / variance
                raw_payoffs = raw_payoffs - beta * (geometric_payoffs - benchmark)
            method = f"{config.method_label} + geometric control variate"
        else:
            method = config.method_label
        benchmark_price = None
        absolute_error = None
    else:
        raw_payoffs = discount * _basket_payoff(geometric_basket, spec.strike, spec.option_type)
        benchmark_price = geometric_basket_price(spec)
        absolute_error = abs(float(np.mean(raw_payoffs)) - benchmark_price)
        method = config.method_label

    runtime_seconds = perf_counter() - start
    price = float(np.mean(raw_payoffs))
    standard_error = float(np.std(raw_payoffs, ddof=1) / math.sqrt(config.num_paths))
    if abs(standard_error) < 1e-15:
        standard_error = 0.0
    margin = 1.96 * standard_error
    if benchmark_price is not None:
        absolute_error = abs(price - benchmark_price)

    return MonteCarloResult(
        price=price,
        standard_error=standard_error,
        confidence_interval=(price - margin, price + margin),
        method=method,
        num_paths=config.num_paths,
        time_steps=1,
        runtime_seconds=runtime_seconds,
        benchmark_price=benchmark_price,
        absolute_error=absolute_error,
        greeks=None,
    )
