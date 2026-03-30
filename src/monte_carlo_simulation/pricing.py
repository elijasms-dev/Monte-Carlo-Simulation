from __future__ import annotations

from dataclasses import dataclass, replace
import math
from time import perf_counter

import numpy as np
from numpy.typing import NDArray

from .sampling import halton_uniforms, inverse_normal_cdf

CALL = "call"
PUT = "put"
EUROPEAN = "european"
ASIAN_ARITHMETIC = "asian-arithmetic"
UP_AND_OUT = "up-and-out"
CONFIDENCE_Z = 1.96

SUPPORTED_OPTION_TYPES = (CALL, PUT)
SUPPORTED_PAYOFFS = (EUROPEAN, ASIAN_ARITHMETIC, UP_AND_OUT)


@dataclass(frozen=True)
class OptionSpec:
    spot: float
    strike: float
    rate: float
    volatility: float
    maturity: float
    option_type: str = CALL
    payoff: str = EUROPEAN
    barrier: float | None = None


@dataclass(frozen=True)
class SimulationConfig:
    num_paths: int = 100_000
    time_steps: int = 252
    seed: int | None = None
    antithetic: bool = True
    control_variate: bool = False
    sampling: str = "pseudo"

    @property
    def method_label(self) -> str:
        labels: list[str] = []
        if self.sampling == "halton":
            labels.append("halton")
        else:
            labels.append("antithetic" if self.antithetic else "naive")
        if self.control_variate:
            labels.append("control variate")
        return " + ".join(labels)


@dataclass(frozen=True)
class GreeksEstimate:
    delta: float
    gamma: float
    vega: float


@dataclass(frozen=True)
class MonteCarloResult:
    price: float
    standard_error: float
    confidence_interval: tuple[float, float]
    method: str
    num_paths: int
    time_steps: int
    runtime_seconds: float
    benchmark_price: float | None
    absolute_error: float | None
    greeks: GreeksEstimate | None = None


@dataclass(frozen=True)
class StudyResult:
    num_paths: int
    method: str
    price: float
    standard_error: float
    confidence_interval: tuple[float, float]
    benchmark_price: float | None
    absolute_error: float | None
    runtime_seconds: float
    efficiency_gain: float | None


def _validate_option_spec(spec: OptionSpec) -> None:
    if spec.spot <= 0:
        raise ValueError("spot must be positive.")
    if spec.strike <= 0:
        raise ValueError("strike must be positive.")
    if spec.volatility < 0:
        raise ValueError("volatility cannot be negative.")
    if spec.maturity < 0:
        raise ValueError("maturity cannot be negative.")
    if spec.option_type not in SUPPORTED_OPTION_TYPES:
        raise ValueError(f"option_type must be one of {SUPPORTED_OPTION_TYPES}.")
    if spec.payoff not in SUPPORTED_PAYOFFS:
        raise ValueError(f"payoff must be one of {SUPPORTED_PAYOFFS}.")
    if spec.payoff == UP_AND_OUT and spec.barrier is None:
        raise ValueError("barrier must be provided for up-and-out options.")
    if spec.barrier is not None and spec.barrier <= 0:
        raise ValueError("barrier must be positive when provided.")


def _validate_config(config: SimulationConfig) -> None:
    if config.num_paths < 2:
        raise ValueError("num_paths must be at least 2.")
    if config.time_steps < 1:
        raise ValueError("time_steps must be at least 1.")
    if config.sampling not in {"pseudo", "halton"}:
        raise ValueError("sampling must be 'pseudo' or 'halton'.")


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _option_payoff(values: NDArray[np.float64], strike: float, option_type: str) -> NDArray[np.float64]:
    if option_type == CALL:
        return np.maximum(values - strike, 0.0)
    return np.maximum(strike - values, 0.0)


def _deterministic_paths(spec: OptionSpec, config: SimulationConfig) -> NDArray[np.float64]:
    if spec.maturity == 0:
        return np.full((config.num_paths, 1), spec.spot, dtype=np.float64)

    monitoring_times = np.linspace(
        spec.maturity / config.time_steps,
        spec.maturity,
        config.time_steps,
        dtype=np.float64,
    )
    deterministic = spec.spot * np.exp(spec.rate * monitoring_times)
    return np.broadcast_to(deterministic, (config.num_paths, config.time_steps)).copy()


def _generate_standard_normals(config: SimulationConfig) -> NDArray[np.float64]:
    base_paths = math.ceil(config.num_paths / 2) if config.antithetic else config.num_paths
    if config.sampling == "halton":
        uniforms = halton_uniforms(base_paths, config.time_steps, seed=config.seed)
        if config.antithetic:
            mirrored = 1.0 - uniforms
            uniforms = np.vstack((uniforms, mirrored))[: config.num_paths]
        return inverse_normal_cdf(uniforms)

    rng = np.random.default_rng(config.seed)
    draws = rng.standard_normal((base_paths, config.time_steps), dtype=np.float64)
    if not config.antithetic:
        return draws
    return np.vstack((draws, -draws))[: config.num_paths]


def _build_paths_from_normals(
    spec: OptionSpec,
    config: SimulationConfig,
    normals: NDArray[np.float64],
) -> NDArray[np.float64]:
    if spec.maturity == 0 or spec.volatility == 0:
        return _deterministic_paths(spec, config)

    dt = spec.maturity / config.time_steps
    drift = (spec.rate - 0.5 * spec.volatility * spec.volatility) * dt
    diffusion = spec.volatility * math.sqrt(dt)
    log_returns = drift + diffusion * normals
    return spec.spot * np.exp(np.cumsum(log_returns, axis=1))


def _discounted_payoffs(
    spec: OptionSpec,
    config: SimulationConfig,
    paths: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    terminal_prices = paths[:, -1]
    discount_factor = math.exp(-spec.rate * spec.maturity)

    if spec.payoff == EUROPEAN:
        raw_payoffs = _option_payoff(terminal_prices, spec.strike, spec.option_type)
    elif spec.payoff == ASIAN_ARITHMETIC:
        arithmetic_average = np.mean(paths, axis=1)
        raw_payoffs = _option_payoff(arithmetic_average, spec.strike, spec.option_type)
    else:
        barrier = float(spec.barrier)
        knocked_out = np.logical_or(np.max(paths, axis=1) >= barrier, spec.spot >= barrier)
        vanilla_payoffs = _option_payoff(terminal_prices, spec.strike, spec.option_type)
        raw_payoffs = np.where(knocked_out, 0.0, vanilla_payoffs)

    return discount_factor * raw_payoffs, terminal_prices


def _apply_control_variate(
    payoffs: NDArray[np.float64],
    terminal_prices: NDArray[np.float64],
    spec: OptionSpec,
) -> NDArray[np.float64]:
    if payoffs.size < 2:
        return payoffs

    discount_factor = math.exp(-spec.rate * spec.maturity)
    control = discount_factor * terminal_prices
    variance = float(np.var(control, ddof=1))
    if variance == 0:
        return payoffs

    covariance = float(np.cov(payoffs, control, ddof=1)[0, 1])
    beta = covariance / variance
    return payoffs - beta * (control - spec.spot)


def _benchmark_price(spec: OptionSpec) -> float | None:
    if spec.payoff != EUROPEAN:
        return None
    return black_scholes_price(
        spot=spec.spot,
        strike=spec.strike,
        rate=spec.rate,
        volatility=spec.volatility,
        maturity=spec.maturity,
        option_type=spec.option_type,
    )


def _summarize_payoffs(
    payoffs: NDArray[np.float64],
    spec: OptionSpec,
    config: SimulationConfig,
    runtime_seconds: float,
    greeks: GreeksEstimate | None,
) -> MonteCarloResult:
    price = float(np.mean(payoffs))
    standard_error = float(np.std(payoffs, ddof=1) / math.sqrt(payoffs.size))
    if abs(standard_error) < 1e-15:
        standard_error = 0.0
    margin = CONFIDENCE_Z * standard_error
    benchmark_price = _benchmark_price(spec)
    absolute_error = (
        abs(price - benchmark_price) if benchmark_price is not None else None
    )
    return MonteCarloResult(
        price=price,
        standard_error=standard_error,
        confidence_interval=(price - margin, price + margin),
        method=config.method_label,
        num_paths=config.num_paths,
        time_steps=config.time_steps,
        runtime_seconds=runtime_seconds,
        benchmark_price=benchmark_price,
        absolute_error=absolute_error,
        greeks=greeks,
    )


def _price_from_normals(
    spec: OptionSpec,
    config: SimulationConfig,
    normals: NDArray[np.float64] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    if normals is None:
        paths = _deterministic_paths(spec, config)
    else:
        paths = _build_paths_from_normals(spec, config, normals)
    payoffs, terminal_prices = _discounted_payoffs(spec, config, paths)
    adjusted_payoffs = (
        _apply_control_variate(payoffs, terminal_prices, spec)
        if config.control_variate
        else payoffs
    )
    return adjusted_payoffs, terminal_prices, paths


def _estimate_greeks(
    spec: OptionSpec,
    config: SimulationConfig,
    normals: NDArray[np.float64] | None,
    base_price: float,
) -> GreeksEstimate:
    spot_shift = max(spec.spot * 0.01, 0.5)
    down_spot = max(spec.spot - spot_shift, 1e-6)
    up_spot = spec.spot + spot_shift

    vol_shift = max(spec.volatility * 0.05, 0.01)
    down_vol = max(spec.volatility - vol_shift, 1e-6)
    up_vol = spec.volatility + vol_shift

    up_spec = replace(spec, spot=up_spot)
    down_spec = replace(spec, spot=down_spot)
    high_vol_spec = replace(spec, volatility=up_vol)
    low_vol_spec = replace(spec, volatility=down_vol)

    up_price = float(np.mean(_price_from_normals(up_spec, config, normals)[0]))
    down_price = float(np.mean(_price_from_normals(down_spec, config, normals)[0]))
    high_vol_price = float(np.mean(_price_from_normals(high_vol_spec, config, normals)[0]))
    low_vol_price = float(np.mean(_price_from_normals(low_vol_spec, config, normals)[0]))

    delta = (up_price - down_price) / (up_spot - down_spot)
    gamma = (up_price - 2.0 * base_price + down_price) / ((0.5 * (up_spot - down_spot)) ** 2)
    vega = (high_vol_price - low_vol_price) / (up_vol - down_vol)
    return GreeksEstimate(delta=delta, gamma=gamma, vega=vega)


def black_scholes_price(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float,
    option_type: str = CALL,
) -> float:
    spec = OptionSpec(
        spot=spot,
        strike=strike,
        rate=rate,
        volatility=volatility,
        maturity=maturity,
        option_type=option_type,
    )
    _validate_option_spec(spec)

    if maturity == 0:
        return float(max(spot - strike, 0.0) if option_type == CALL else max(strike - spot, 0.0))

    discount_factor = math.exp(-rate * maturity)
    if volatility == 0:
        forward_terminal = spot * math.exp(rate * maturity)
        intrinsic = (
            max(forward_terminal - strike, 0.0)
            if option_type == CALL
            else max(strike - forward_terminal, 0.0)
        )
        return float(discount_factor * intrinsic)

    sqrt_maturity = math.sqrt(maturity)
    d1 = (
        math.log(spot / strike)
        + (rate + 0.5 * volatility * volatility) * maturity
    ) / (volatility * sqrt_maturity)
    d2 = d1 - volatility * sqrt_maturity

    if option_type == CALL:
        return float(spot * _normal_cdf(d1) - strike * discount_factor * _normal_cdf(d2))
    return float(strike * discount_factor * _normal_cdf(-d2) - spot * _normal_cdf(-d1))


def black_scholes_delta(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float,
    option_type: str = CALL,
) -> float:
    if maturity == 0:
        if option_type == CALL:
            return 1.0 if spot > strike else 0.0
        return -1.0 if spot < strike else 0.0
    if volatility == 0:
        forward_terminal = spot * math.exp(rate * maturity)
        if option_type == CALL:
            return 1.0 if forward_terminal > strike else 0.0
        return -1.0 if forward_terminal < strike else 0.0

    sqrt_maturity = math.sqrt(maturity)
    d1 = (
        math.log(spot / strike)
        + (rate + 0.5 * volatility * volatility) * maturity
    ) / (volatility * sqrt_maturity)
    if option_type == CALL:
        return float(_normal_cdf(d1))
    return float(_normal_cdf(d1) - 1.0)


def black_scholes_gamma(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float,
) -> float:
    if maturity == 0 or volatility == 0:
        return 0.0
    sqrt_maturity = math.sqrt(maturity)
    d1 = (
        math.log(spot / strike)
        + (rate + 0.5 * volatility * volatility) * maturity
    ) / (volatility * sqrt_maturity)
    density = math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)
    return float(density / (spot * volatility * sqrt_maturity))


def black_scholes_vega(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float,
) -> float:
    if maturity == 0 or volatility == 0:
        return 0.0
    sqrt_maturity = math.sqrt(maturity)
    d1 = (
        math.log(spot / strike)
        + (rate + 0.5 * volatility * volatility) * maturity
    ) / (volatility * sqrt_maturity)
    density = math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)
    return float(spot * density * sqrt_maturity)


def price_option_mc(
    spec: OptionSpec,
    config: SimulationConfig = SimulationConfig(),
    include_greeks: bool | None = None,
) -> MonteCarloResult:
    _validate_option_spec(spec)
    _validate_config(config)

    if include_greeks is None:
        include_greeks = spec.payoff == EUROPEAN

    start = perf_counter()
    normals = None if spec.maturity == 0 or spec.volatility == 0 else _generate_standard_normals(config)
    payoffs, _, _ = _price_from_normals(spec, config, normals)
    runtime_seconds = perf_counter() - start

    greeks = None
    if include_greeks and spec.payoff == EUROPEAN:
        greeks = _estimate_greeks(spec, config, normals, float(np.mean(payoffs)))

    return _summarize_payoffs(
        payoffs=payoffs,
        spec=spec,
        config=config,
        runtime_seconds=runtime_seconds,
        greeks=greeks,
    )


def price_european_option_mc(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float,
    num_paths: int = 100_000,
    option_type: str = CALL,
    seed: int | None = None,
    antithetic: bool = True,
    control_variate: bool = False,
    time_steps: int = 1,
) -> MonteCarloResult:
    spec = OptionSpec(
        spot=spot,
        strike=strike,
        rate=rate,
        volatility=volatility,
        maturity=maturity,
        option_type=option_type,
        payoff=EUROPEAN,
    )
    config = SimulationConfig(
        num_paths=num_paths,
        time_steps=time_steps,
        seed=seed,
        antithetic=antithetic,
        control_variate=control_variate,
    )
    return price_option_mc(spec=spec, config=config, include_greeks=True)


def build_method_config(
    method: str,
    *,
    num_paths: int,
    time_steps: int,
    seed: int | None,
) -> SimulationConfig:
    normalized = method.strip().lower()
    if normalized == "naive":
        return SimulationConfig(
            num_paths=num_paths,
            time_steps=time_steps,
            seed=seed,
            antithetic=False,
            control_variate=False,
            sampling="pseudo",
        )
    if normalized == "antithetic":
        return SimulationConfig(
            num_paths=num_paths,
            time_steps=time_steps,
            seed=seed,
            antithetic=True,
            control_variate=False,
            sampling="pseudo",
        )
    if normalized in {"antithetic-control-variate", "antithetic + control variate"}:
        return SimulationConfig(
            num_paths=num_paths,
            time_steps=time_steps,
            seed=seed,
            antithetic=True,
            control_variate=True,
            sampling="pseudo",
        )
    if normalized == "halton":
        return SimulationConfig(
            num_paths=num_paths,
            time_steps=time_steps,
            seed=seed,
            antithetic=False,
            control_variate=False,
            sampling="halton",
        )
    if normalized in {"halton-control-variate", "halton + control variate"}:
        return SimulationConfig(
            num_paths=num_paths,
            time_steps=time_steps,
            seed=seed,
            antithetic=False,
            control_variate=True,
            sampling="halton",
        )
    raise ValueError(
        "method must be one of: naive, antithetic, antithetic-control-variate, halton, halton-control-variate."
    )
