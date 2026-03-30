from __future__ import annotations

from dataclasses import dataclass
import math
from time import perf_counter

import numpy as np
from numpy.typing import NDArray

from .pricing import CALL, EUROPEAN, OptionSpec, SimulationConfig, _validate_config, _validate_option_spec, black_scholes_price


@dataclass(frozen=True)
class AmericanOptionResult:
    price: float
    standard_error: float
    confidence_interval: tuple[float, float]
    method: str
    num_paths: int
    time_steps: int
    runtime_seconds: float
    early_exercise_ratio: float
    european_reference_price: float
    premium_over_european: float


def _option_payoff(
    prices: NDArray[np.float64] | float,
    strike: float,
    option_type: str,
) -> NDArray[np.float64] | float:
    if option_type == CALL:
        return np.maximum(prices - strike, 0.0)
    return np.maximum(strike - prices, 0.0)


def _generate_standard_normals(config: SimulationConfig) -> NDArray[np.float64]:
    rng = np.random.default_rng(config.seed)
    base_paths = math.ceil(config.num_paths / 2) if config.antithetic else config.num_paths
    draws = rng.standard_normal((base_paths, config.time_steps), dtype=np.float64)
    if not config.antithetic:
        return draws
    return np.vstack((draws, -draws))[: config.num_paths]


def _simulate_paths(spec: OptionSpec, config: SimulationConfig) -> NDArray[np.float64]:
    normals = _generate_standard_normals(config)
    dt = spec.maturity / config.time_steps
    drift = (spec.rate - 0.5 * spec.volatility * spec.volatility) * dt
    diffusion = spec.volatility * math.sqrt(dt)
    log_increments = drift + diffusion * normals
    paths = np.empty((config.num_paths, config.time_steps + 1), dtype=np.float64)
    paths[:, 0] = spec.spot
    paths[:, 1:] = spec.spot * np.exp(np.cumsum(log_increments, axis=1))
    return paths


def _deterministic_american_price(spec: OptionSpec) -> tuple[float, float]:
    immediate = float(_option_payoff(spec.spot, spec.strike, spec.option_type))
    terminal_spot = spec.spot * math.exp(spec.rate * spec.maturity)
    discounted_terminal = math.exp(-spec.rate * spec.maturity) * float(
        _option_payoff(terminal_spot, spec.strike, spec.option_type)
    )
    if spec.option_type == CALL:
        return discounted_terminal, 0.0
    if immediate >= discounted_terminal:
        return immediate, 1.0 if immediate > 0.0 else 0.0
    return discounted_terminal, 0.0


def price_american_option_lsm(
    spec: OptionSpec,
    config: SimulationConfig = SimulationConfig(num_paths=50_000, time_steps=50),
    *,
    basis_order: int = 2,
) -> AmericanOptionResult:
    _validate_option_spec(spec)
    _validate_config(config)
    if spec.payoff != EUROPEAN:
        raise ValueError("American pricing currently supports vanilla call/put payoffs only.")

    european_reference_price = black_scholes_price(
        spot=spec.spot,
        strike=spec.strike,
        rate=spec.rate,
        volatility=spec.volatility,
        maturity=spec.maturity,
        option_type=spec.option_type,
    )

    if spec.option_type == CALL:
        return AmericanOptionResult(
            price=european_reference_price,
            standard_error=0.0,
            confidence_interval=(european_reference_price, european_reference_price),
            method="no-early-exercise identity",
            num_paths=config.num_paths,
            time_steps=config.time_steps,
            runtime_seconds=0.0,
            early_exercise_ratio=0.0,
            european_reference_price=european_reference_price,
            premium_over_european=0.0,
        )

    if spec.maturity == 0 or spec.volatility == 0:
        price, early_exercise_ratio = _deterministic_american_price(spec)
        return AmericanOptionResult(
            price=price,
            standard_error=0.0,
            confidence_interval=(price, price),
            method=f"deterministic policy (poly order {basis_order})",
            num_paths=config.num_paths,
            time_steps=config.time_steps,
            runtime_seconds=0.0,
            early_exercise_ratio=early_exercise_ratio,
            european_reference_price=european_reference_price,
            premium_over_european=price - european_reference_price,
        )

    start = perf_counter()
    paths = _simulate_paths(spec, config)
    dt = spec.maturity / config.time_steps
    discount = math.exp(-spec.rate * dt)
    cashflows = _option_payoff(paths[:, -1], spec.strike, spec.option_type).astype(np.float64)
    exercise_times = np.full(config.num_paths, config.time_steps, dtype=np.int32)

    for step in range(config.time_steps - 1, 0, -1):
        cashflows *= discount
        spot_t = paths[:, step]
        exercise_values = _option_payoff(spot_t, spec.strike, spec.option_type)
        in_the_money = exercise_values > 0.0
        if np.count_nonzero(in_the_money) <= basis_order + 1:
            continue

        x = spot_t[in_the_money]
        y = cashflows[in_the_money]
        basis = np.column_stack([x**power for power in range(basis_order + 1)])
        coefficients, *_ = np.linalg.lstsq(basis, y, rcond=None)
        continuation = basis @ coefficients
        should_exercise = exercise_values[in_the_money] > continuation
        if not np.any(should_exercise):
            continue

        exercise_indices = np.flatnonzero(in_the_money)[should_exercise]
        cashflows[exercise_indices] = exercise_values[exercise_indices]
        exercise_times[exercise_indices] = step

    present_values = cashflows * discount
    runtime_seconds = perf_counter() - start
    price = float(np.mean(present_values))
    standard_error = float(np.std(present_values, ddof=1) / math.sqrt(config.num_paths))
    if abs(standard_error) < 1e-15:
        standard_error = 0.0
    margin = 1.96 * standard_error

    return AmericanOptionResult(
        price=price,
        standard_error=standard_error,
        confidence_interval=(price - margin, price + margin),
        method=f"Longstaff-Schwartz (poly order {basis_order})",
        num_paths=config.num_paths,
        time_steps=config.time_steps,
        runtime_seconds=runtime_seconds,
        early_exercise_ratio=float(np.mean(exercise_times < config.time_steps)),
        european_reference_price=european_reference_price,
        premium_over_european=price - european_reference_price,
    )

