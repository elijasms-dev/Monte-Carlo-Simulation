from __future__ import annotations

from dataclasses import dataclass
import csv
import math
from pathlib import Path
from time import perf_counter

import numpy as np
from numpy.typing import NDArray

from .pricing import CALL, PUT, SimulationConfig, _validate_config
from .sampling import halton_uniforms, inverse_normal_cdf


CONFIDENCE_Z = 1.96


@dataclass(frozen=True)
class HestonSpec:
    spot: float
    strike: float
    rate: float
    maturity: float
    initial_variance: float
    long_run_variance: float
    mean_reversion: float
    vol_of_vol: float
    correlation: float
    option_type: str = CALL
    dividend_yield: float = 0.0


@dataclass(frozen=True)
class HestonResult:
    price: float
    standard_error: float
    confidence_interval: tuple[float, float]
    method: str
    num_paths: int
    time_steps: int
    runtime_seconds: float
    implied_volatility: float | None
    mean_terminal_variance: float
    mean_path_variance: float


@dataclass(frozen=True)
class HestonSmilePoint:
    strike: float
    moneyness: float
    price: float
    standard_error: float
    implied_volatility: float | None


def _validate_heston_spec(spec: HestonSpec) -> None:
    if spec.spot <= 0:
        raise ValueError("spot must be positive.")
    if spec.strike <= 0:
        raise ValueError("strike must be positive.")
    if spec.maturity < 0:
        raise ValueError("maturity cannot be negative.")
    if spec.initial_variance < 0:
        raise ValueError("initial_variance cannot be negative.")
    if spec.long_run_variance < 0:
        raise ValueError("long_run_variance cannot be negative.")
    if spec.mean_reversion < 0:
        raise ValueError("mean_reversion cannot be negative.")
    if spec.vol_of_vol < 0:
        raise ValueError("vol_of_vol cannot be negative.")
    if abs(spec.correlation) > 1.0:
        raise ValueError("correlation must be between -1 and 1.")
    if spec.dividend_yield < 0:
        raise ValueError("dividend_yield cannot be negative.")
    if spec.option_type not in {CALL, PUT}:
        raise ValueError("option_type must be 'call' or 'put'.")


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _black_scholes_price_with_yield(
    spot: float,
    strike: float,
    rate: float,
    maturity: float,
    volatility: float,
    option_type: str,
    dividend_yield: float = 0.0,
) -> float:
    if maturity == 0:
        return float(max(spot - strike, 0.0) if option_type == CALL else max(strike - spot, 0.0))

    discount_factor = math.exp(-rate * maturity)
    dividend_discount = math.exp(-dividend_yield * maturity)
    if volatility == 0:
        forward_terminal = spot * math.exp((rate - dividend_yield) * maturity)
        intrinsic = (
            max(forward_terminal - strike, 0.0)
            if option_type == CALL
            else max(strike - forward_terminal, 0.0)
        )
        return float(discount_factor * intrinsic)

    sqrt_maturity = math.sqrt(maturity)
    d1 = (
        math.log(spot / strike)
        + (rate - dividend_yield + 0.5 * volatility * volatility) * maturity
    ) / (volatility * sqrt_maturity)
    d2 = d1 - volatility * sqrt_maturity

    if option_type == CALL:
        return float(spot * dividend_discount * _normal_cdf(d1) - strike * discount_factor * _normal_cdf(d2))
    return float(strike * discount_factor * _normal_cdf(-d2) - spot * dividend_discount * _normal_cdf(-d1))


def implied_volatility_from_price(
    price: float,
    *,
    spot: float,
    strike: float,
    rate: float,
    maturity: float,
    option_type: str = CALL,
    dividend_yield: float = 0.0,
    tolerance: float = 1e-8,
    max_iterations: int = 200,
) -> float | None:
    if maturity == 0:
        return 0.0

    low = 1e-6
    high = 1.0
    low_price = _black_scholes_price_with_yield(
        spot, strike, rate, maturity, low, option_type, dividend_yield
    )
    if price <= low_price + tolerance:
        return 0.0

    high_price = _black_scholes_price_with_yield(
        spot, strike, rate, maturity, high, option_type, dividend_yield
    )
    while high_price < price and high < 10.0:
        high *= 2.0
        high_price = _black_scholes_price_with_yield(
            spot, strike, rate, maturity, high, option_type, dividend_yield
        )
    if high_price < price:
        return None

    for _ in range(max_iterations):
        mid = 0.5 * (low + high)
        mid_price = _black_scholes_price_with_yield(
            spot, strike, rate, maturity, mid, option_type, dividend_yield
        )
        if abs(mid_price - price) < tolerance:
            return mid
        if mid_price < price:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def _option_payoff(values: NDArray[np.float64], strike: float, option_type: str) -> NDArray[np.float64]:
    if option_type == CALL:
        return np.maximum(values - strike, 0.0)
    return np.maximum(strike - values, 0.0)


def _generate_heston_normals(config: SimulationConfig) -> NDArray[np.float64]:
    base_paths = math.ceil(config.num_paths / 2) if config.antithetic else config.num_paths
    dimension = 2 * config.time_steps
    if config.sampling == "halton":
        uniforms = halton_uniforms(base_paths, dimension, seed=config.seed)
        if config.antithetic:
            uniforms = np.vstack((uniforms, 1.0 - uniforms))[: config.num_paths]
        normals = inverse_normal_cdf(uniforms)
    else:
        rng = np.random.default_rng(config.seed)
        normals = rng.standard_normal((base_paths, dimension), dtype=np.float64)
        if config.antithetic:
            normals = np.vstack((normals, -normals))[: config.num_paths]
    return normals.reshape(config.num_paths, config.time_steps, 2)


def _simulate_heston_paths(
    spec: HestonSpec,
    config: SimulationConfig,
) -> tuple[NDArray[np.float64], float, float]:
    if spec.maturity == 0:
        terminal = np.full(config.num_paths, spec.spot, dtype=np.float64)
        return terminal, spec.initial_variance, spec.initial_variance

    normals = _generate_heston_normals(config)
    spots = np.full(config.num_paths, spec.spot, dtype=np.float64)
    variances = np.full(config.num_paths, spec.initial_variance, dtype=np.float64)
    variance_running_sum = np.zeros(config.num_paths, dtype=np.float64)

    dt = spec.maturity / config.time_steps
    sqrt_dt = math.sqrt(dt)
    corr_scale = math.sqrt(max(1.0 - spec.correlation * spec.correlation, 0.0))

    for step in range(config.time_steps):
        z_spot = normals[:, step, 0]
        z_var = spec.correlation * z_spot + corr_scale * normals[:, step, 1]
        positive_variance = np.maximum(variances, 0.0)
        variance_running_sum += positive_variance

        spots *= np.exp(
            (spec.rate - spec.dividend_yield - 0.5 * positive_variance) * dt
            + np.sqrt(positive_variance) * sqrt_dt * z_spot
        )
        variances = (
            variances
            + spec.mean_reversion * (spec.long_run_variance - positive_variance) * dt
            + spec.vol_of_vol * np.sqrt(positive_variance) * sqrt_dt * z_var
        )
        variances = np.maximum(variances, 0.0)

    return (
        spots,
        float(np.mean(variances)),
        float(np.mean(variance_running_sum / config.time_steps)),
    )


def _heston_characteristic_function(
    spec: HestonSpec,
    u: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    x0 = math.log(spec.spot)
    iu = 1j * u
    kappa = spec.mean_reversion
    theta = spec.long_run_variance
    sigma = spec.vol_of_vol
    rho = spec.correlation
    v0 = spec.initial_variance
    tau = spec.maturity

    if tau == 0:
        return np.exp(iu * x0)

    if sigma == 0:
        total_variance = max(v0, 0.0)
        drift = x0 + (spec.rate - spec.dividend_yield - 0.5 * total_variance) * tau
        diffusion = total_variance * tau
        return np.exp(iu * drift - 0.5 * u * u * diffusion)

    beta = kappa - rho * sigma * iu
    d = np.sqrt(beta * beta + sigma * sigma * (u * u + iu))
    g = (beta - d) / (beta + d)
    exp_dt = np.exp(-d * tau)
    one_minus_g_exp = 1.0 - g * exp_dt
    one_minus_g = 1.0 - g

    c_term = (
        iu * (x0 + (spec.rate - spec.dividend_yield) * tau)
        + (kappa * theta / (sigma * sigma))
        * ((beta - d) * tau - 2.0 * np.log(one_minus_g_exp / one_minus_g))
    )
    d_term = ((beta - d) / (sigma * sigma)) * ((1.0 - exp_dt) / one_minus_g_exp)
    return np.exp(c_term + d_term * v0)


def price_heston_option_cf(
    spec: HestonSpec,
    *,
    integration_limit: float = 150.0,
    integration_points: int = 1_024,
) -> float:
    _validate_heston_spec(spec)
    if spec.maturity == 0:
        return float(max(spec.spot - spec.strike, 0.0) if spec.option_type == CALL else max(spec.strike - spec.spot, 0.0))

    if integration_points < 50:
        raise ValueError("integration_points must be at least 50.")

    u = np.linspace(1e-8, integration_limit, integration_points, dtype=np.float64)
    complex_u = u.astype(np.complex128)
    log_strike = math.log(spec.strike)

    cf_u = _heston_characteristic_function(spec, complex_u)
    cf_shifted = _heston_characteristic_function(spec, complex_u - 1j)
    forward_factor = spec.spot * math.exp((spec.rate - spec.dividend_yield) * spec.maturity)

    integrand_p1 = np.real(
        np.exp(-1j * complex_u * log_strike)
        * cf_shifted
        / (1j * complex_u * forward_factor)
    )
    integrand_p2 = np.real(
        np.exp(-1j * complex_u * log_strike) * cf_u / (1j * complex_u)
    )
    p1 = 0.5 + np.trapezoid(integrand_p1, u) / math.pi
    p2 = 0.5 + np.trapezoid(integrand_p2, u) / math.pi

    discount = math.exp(-spec.rate * spec.maturity)
    dividend_discount = math.exp(-spec.dividend_yield * spec.maturity)
    call_price = spec.spot * dividend_discount * p1 - spec.strike * discount * p2
    if spec.option_type == CALL:
        return float(max(call_price, 0.0))

    put_price = call_price - spec.spot * dividend_discount + spec.strike * discount
    return float(max(put_price, 0.0))


def price_heston_option_mc(
    spec: HestonSpec,
    config: SimulationConfig = SimulationConfig(num_paths=100_000, time_steps=128),
) -> HestonResult:
    _validate_heston_spec(spec)
    _validate_config(config)

    start = perf_counter()
    terminal_spots, mean_terminal_variance, mean_path_variance = _simulate_heston_paths(spec, config)
    discount_factor = math.exp(-spec.rate * spec.maturity)
    payoffs = discount_factor * _option_payoff(terminal_spots, spec.strike, spec.option_type)
    runtime_seconds = perf_counter() - start

    price = float(np.mean(payoffs))
    standard_error = float(np.std(payoffs, ddof=1) / math.sqrt(config.num_paths))
    if abs(standard_error) < 1e-15:
        standard_error = 0.0
    margin = CONFIDENCE_Z * standard_error
    implied_volatility = implied_volatility_from_price(
        price,
        spot=spec.spot,
        strike=spec.strike,
        rate=spec.rate,
        maturity=spec.maturity,
        option_type=spec.option_type,
        dividend_yield=spec.dividend_yield,
    )

    return HestonResult(
        price=price,
        standard_error=standard_error,
        confidence_interval=(price - margin, price + margin),
        method=f"Heston full-truncation Euler ({config.method_label})",
        num_paths=config.num_paths,
        time_steps=config.time_steps,
        runtime_seconds=runtime_seconds,
        implied_volatility=implied_volatility,
        mean_terminal_variance=mean_terminal_variance,
        mean_path_variance=mean_path_variance,
    )


def run_heston_smile(
    spec: HestonSpec,
    strikes: list[float],
    config: SimulationConfig = SimulationConfig(num_paths=100_000, time_steps=128),
) -> list[HestonSmilePoint]:
    _validate_heston_spec(spec)
    _validate_config(config)
    if any(strike <= 0 for strike in strikes):
        raise ValueError("all strikes must be positive.")

    terminal_spots, _, _ = _simulate_heston_paths(spec, config)
    discount_factor = math.exp(-spec.rate * spec.maturity)
    smile: list[HestonSmilePoint] = []
    for strike in strikes:
        payoffs = discount_factor * _option_payoff(terminal_spots, strike, spec.option_type)
        price = float(np.mean(payoffs))
        standard_error = float(np.std(payoffs, ddof=1) / math.sqrt(config.num_paths))
        if abs(standard_error) < 1e-15:
            standard_error = 0.0
        implied_volatility = implied_volatility_from_price(
            price,
            spot=spec.spot,
            strike=strike,
            rate=spec.rate,
            maturity=spec.maturity,
            option_type=spec.option_type,
            dividend_yield=spec.dividend_yield,
        )
        smile.append(
            HestonSmilePoint(
                strike=strike,
                moneyness=strike / spec.spot,
                price=price,
                standard_error=standard_error,
                implied_volatility=implied_volatility,
            )
        )
    return smile


def run_heston_smile_cf(
    spec: HestonSpec,
    strikes: list[float],
    *,
    integration_limit: float = 150.0,
    integration_points: int = 1_024,
) -> list[HestonSmilePoint]:
    _validate_heston_spec(spec)
    if any(strike <= 0 for strike in strikes):
        raise ValueError("all strikes must be positive.")

    smile: list[HestonSmilePoint] = []
    for strike in strikes:
        strike_spec = HestonSpec(
            spot=spec.spot,
            strike=strike,
            rate=spec.rate,
            maturity=spec.maturity,
            initial_variance=spec.initial_variance,
            long_run_variance=spec.long_run_variance,
            mean_reversion=spec.mean_reversion,
            vol_of_vol=spec.vol_of_vol,
            correlation=spec.correlation,
            option_type=spec.option_type,
            dividend_yield=spec.dividend_yield,
        )
        price = price_heston_option_cf(
            strike_spec,
            integration_limit=integration_limit,
            integration_points=integration_points,
        )
        implied_volatility = implied_volatility_from_price(
            price,
            spot=spec.spot,
            strike=strike,
            rate=spec.rate,
            maturity=spec.maturity,
            option_type=spec.option_type,
            dividend_yield=spec.dividend_yield,
        )
        smile.append(
            HestonSmilePoint(
                strike=strike,
                moneyness=strike / spec.spot,
                price=price,
                standard_error=0.0,
                implied_volatility=implied_volatility,
            )
        )
    return smile


def format_heston_smile_table(points: list[HestonSmilePoint]) -> str:
    headers = (
        ("strike", 10),
        ("moneyness", 11),
        ("price", 11),
        ("stderr", 11),
        ("impl vol", 11),
    )
    lines = [" ".join(label.ljust(width) for label, width in headers)]
    lines.append(" ".join("-" * width for _, width in headers))
    for point in points:
        implied_vol = "-" if point.implied_volatility is None else f"{point.implied_volatility:.5f}"
        values = (
            f"{point.strike:.2f}",
            f"{point.moneyness:.3f}",
            f"{point.price:.5f}",
            f"{point.standard_error:.5f}",
            implied_vol,
        )
        lines.append(
            " ".join(value.ljust(width) for value, (_, width) in zip(values, headers))
        )
    return "\n".join(lines)


def write_heston_smile_csv(points: list[HestonSmilePoint], destination: str) -> Path:
    output_path = Path(destination)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["strike", "moneyness", "price", "standard_error", "implied_volatility"])
        for point in points:
            writer.writerow(
                [
                    point.strike,
                    point.moneyness,
                    point.price,
                    point.standard_error,
                    point.implied_volatility,
                ]
            )
    return output_path
