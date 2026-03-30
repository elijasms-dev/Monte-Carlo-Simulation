from __future__ import annotations

from dataclasses import dataclass
import math
import random
from statistics import fmean

CALL = "call"
PUT = "put"
CONFIDENCE_Z = 1.96


@dataclass(frozen=True)
class MonteCarloResult:
    price: float
    standard_error: float
    confidence_interval: tuple[float, float]


def _validate_inputs(
    spot: float,
    strike: float,
    volatility: float,
    maturity: float,
    num_paths: int,
    option_type: str,
) -> None:
    if spot <= 0:
        raise ValueError("spot must be positive.")
    if strike <= 0:
        raise ValueError("strike must be positive.")
    if volatility < 0:
        raise ValueError("volatility cannot be negative.")
    if maturity < 0:
        raise ValueError("maturity cannot be negative.")
    if num_paths < 2:
        raise ValueError("num_paths must be at least 2.")
    if option_type not in {CALL, PUT}:
        raise ValueError("option_type must be 'call' or 'put'.")


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _discounted_intrinsic_value(
    spot: float,
    strike: float,
    rate: float,
    maturity: float,
    option_type: str,
) -> float:
    terminal_price = spot * math.exp(rate * maturity)
    intrinsic = (
        max(terminal_price - strike, 0.0)
        if option_type == CALL
        else max(strike - terminal_price, 0.0)
    )
    return math.exp(-rate * maturity) * intrinsic


def black_scholes_price(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float,
    option_type: str = CALL,
) -> float:
    _validate_inputs(
        spot=spot,
        strike=strike,
        volatility=volatility,
        maturity=maturity,
        num_paths=2,
        option_type=option_type,
    )

    if maturity == 0:
        return max(spot - strike, 0.0) if option_type == CALL else max(strike - spot, 0.0)
    if volatility == 0:
        return _discounted_intrinsic_value(spot, strike, rate, maturity, option_type)

    sqrt_maturity = math.sqrt(maturity)
    d1 = (
        math.log(spot / strike)
        + (rate + 0.5 * volatility * volatility) * maturity
    ) / (volatility * sqrt_maturity)
    d2 = d1 - volatility * sqrt_maturity

    if option_type == CALL:
        return spot * _normal_cdf(d1) - strike * math.exp(-rate * maturity) * _normal_cdf(d2)
    return strike * math.exp(-rate * maturity) * _normal_cdf(-d2) - spot * _normal_cdf(-d1)


def _option_payoff(terminal_price: float, strike: float, option_type: str) -> float:
    if option_type == CALL:
        return max(terminal_price - strike, 0.0)
    return max(strike - terminal_price, 0.0)


def _discounted_payoffs(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float,
    num_paths: int,
    option_type: str,
    seed: int | None,
    antithetic: bool,
) -> list[float]:
    rng = random.Random(seed)
    drift = (rate - 0.5 * volatility * volatility) * maturity
    diffusion = volatility * math.sqrt(maturity)
    discount_factor = math.exp(-rate * maturity)
    payoffs: list[float] = []

    while len(payoffs) < num_paths:
        z = rng.gauss(0.0, 1.0)
        draws = (z, -z) if antithetic else (z,)
        for draw in draws:
            terminal_price = spot * math.exp(drift + diffusion * draw)
            payoffs.append(
                discount_factor * _option_payoff(terminal_price, strike, option_type)
            )
            if len(payoffs) == num_paths:
                break

    return payoffs


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
) -> MonteCarloResult:
    _validate_inputs(
        spot=spot,
        strike=strike,
        volatility=volatility,
        maturity=maturity,
        num_paths=num_paths,
        option_type=option_type,
    )

    if maturity == 0 or volatility == 0:
        if maturity == 0:
            deterministic_price = (
                max(spot - strike, 0.0)
                if option_type == CALL
                else max(strike - spot, 0.0)
            )
        else:
            deterministic_price = _discounted_intrinsic_value(
                spot, strike, rate, maturity, option_type
            )
        return MonteCarloResult(
            price=deterministic_price,
            standard_error=0.0,
            confidence_interval=(deterministic_price, deterministic_price),
        )

    payoffs = _discounted_payoffs(
        spot=spot,
        strike=strike,
        rate=rate,
        volatility=volatility,
        maturity=maturity,
        num_paths=num_paths,
        option_type=option_type,
        seed=seed,
        antithetic=antithetic,
    )
    price = fmean(payoffs)
    squared_diff_sum = sum((payoff - price) ** 2 for payoff in payoffs)
    sample_variance = squared_diff_sum / (num_paths - 1)
    standard_error = math.sqrt(sample_variance / num_paths)
    margin = CONFIDENCE_Z * standard_error
    return MonteCarloResult(
        price=price,
        standard_error=standard_error,
        confidence_interval=(price - margin, price + margin),
    )
