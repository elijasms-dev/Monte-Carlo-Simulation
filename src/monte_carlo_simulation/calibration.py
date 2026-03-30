from __future__ import annotations

from dataclasses import dataclass
import csv
from datetime import date
import math
from pathlib import Path
from time import perf_counter

import numpy as np

from .heston import HestonSpec, implied_volatility_from_price, price_heston_option_cf
from .pricing import CALL, PUT


DEFAULT_MARKET_DATASET = "spy_2026-03-11_calls.csv"
DEFAULT_DIVIDEND_YIELD = 0.015


@dataclass(frozen=True)
class HestonParameters:
    initial_variance: float
    long_run_variance: float
    mean_reversion: float
    vol_of_vol: float
    correlation: float


@dataclass(frozen=True)
class MarketOptionQuote:
    snapshot_date: str
    expiration: str
    underlying: str
    spot: float
    strike: float
    maturity: float
    rate: float
    dividend_yield: float
    option_type: str
    bid: float
    ask: float
    market_price: float
    market_implied_volatility: float | None


@dataclass(frozen=True)
class CalibrationPoint:
    expiration: str
    strike: float
    maturity: float
    market_price: float
    model_price: float
    absolute_price_error: float
    market_implied_volatility: float | None
    model_implied_volatility: float | None
    absolute_iv_error: float | None


@dataclass(frozen=True)
class HestonCalibrationResult:
    parameters: HestonParameters
    objective_value: float
    rmse_price: float
    rmse_implied_volatility: float | None
    max_abs_price_error: float
    num_quotes: int
    evaluations: int
    runtime_seconds: float
    fitted_points: tuple[CalibrationPoint, ...]


@dataclass(frozen=True)
class _CalibrationScore:
    parameters: HestonParameters
    objective_value: float
    rmse_price: float
    rmse_implied_volatility: float | None
    max_abs_price_error: float
    fitted_points: tuple[CalibrationPoint, ...]


def bundled_market_data_path(filename: str = DEFAULT_MARKET_DATASET) -> Path:
    return Path(__file__).resolve().parent / "data" / "market" / filename


def load_market_quotes_csv(path: str | Path) -> list[MarketOptionQuote]:
    csv_path = Path(path)
    if not csv_path.exists():
        raise ValueError(f"market data file does not exist: {csv_path}")

    quotes: list[MarketOptionQuote] = []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            snapshot_date = row["snapshot_date"].strip()
            expiration = row["expiration"].strip()
            maturity = _compute_maturity(snapshot_date, expiration)
            option_type = row["option_type"].strip().lower()
            if option_type not in {CALL, PUT}:
                raise ValueError("option_type must be 'call' or 'put'.")

            bid = float(row["bid"])
            ask = float(row["ask"])
            market_price = (
                float(row["mid_price"])
                if row.get("mid_price")
                else 0.5 * (bid + ask)
            )
            spot = float(row["spot"])
            strike = float(row["strike"])
            rate = float(row["rate"])
            dividend_yield = float(row.get("dividend_yield") or DEFAULT_DIVIDEND_YIELD)
            market_iv = implied_volatility_from_price(
                market_price,
                spot=spot,
                strike=strike,
                rate=rate,
                maturity=maturity,
                option_type=option_type,
                dividend_yield=dividend_yield,
            )
            quotes.append(
                MarketOptionQuote(
                    snapshot_date=snapshot_date,
                    expiration=expiration,
                    underlying=row["underlying"].strip(),
                    spot=spot,
                    strike=strike,
                    maturity=maturity,
                    rate=rate,
                    dividend_yield=dividend_yield,
                    option_type=option_type,
                    bid=bid,
                    ask=ask,
                    market_price=market_price,
                    market_implied_volatility=market_iv,
                )
            )

    if not quotes:
        raise ValueError("market data file did not contain any quotes.")
    return sorted(quotes, key=lambda quote: (quote.expiration, quote.strike))


def calibrate_heston_parameters(
    quotes: list[MarketOptionQuote],
    *,
    initial_guess: HestonParameters | None = None,
    seed: int = 7,
    global_samples: int = 48,
    search_rounds: int = 6,
    local_samples: int = 24,
    integration_limit: float = 150.0,
    integration_points: int = 256,
) -> HestonCalibrationResult:
    if len(quotes) < 4:
        raise ValueError("at least four option quotes are required for calibration.")
    if search_rounds <= 0:
        raise ValueError("search_rounds must be positive.")
    if global_samples < 0 or local_samples <= 0:
        raise ValueError("global_samples must be non-negative and local_samples must be positive.")

    bounds = {
        "initial_variance": (0.005, 0.20),
        "long_run_variance": (0.005, 0.20),
        "mean_reversion": (0.25, 8.0),
        "vol_of_vol": (0.05, 1.80),
        "correlation": (-0.95, 0.20),
    }
    guess = initial_guess or HestonParameters(
        initial_variance=0.04,
        long_run_variance=0.04,
        mean_reversion=1.5,
        vol_of_vol=0.50,
        correlation=-0.60,
    )

    start = perf_counter()
    rng = np.random.default_rng(seed)
    evaluations = 0
    best_score = _evaluate_candidate(
        _clip_parameters(guess, bounds),
        quotes,
        integration_limit=integration_limit,
        integration_points=integration_points,
    )
    evaluations += 1

    for _ in range(global_samples):
        score = _evaluate_candidate(
            _sample_parameters(rng, bounds),
            quotes,
            integration_limit=integration_limit,
            integration_points=integration_points,
        )
        evaluations += 1
        if score.objective_value < best_score.objective_value:
            best_score = score

    scales = np.geomspace(0.55, 0.05, search_rounds)
    for scale in scales:
        for candidate in _coordinate_candidates(best_score.parameters, scale, bounds):
            score = _evaluate_candidate(
                candidate,
                quotes,
                integration_limit=integration_limit,
                integration_points=integration_points,
            )
            evaluations += 1
            if score.objective_value < best_score.objective_value:
                best_score = score

        for _ in range(local_samples):
            score = _evaluate_candidate(
                _perturb_parameters(best_score.parameters, rng, scale, bounds),
                quotes,
                integration_limit=integration_limit,
                integration_points=integration_points,
            )
            evaluations += 1
            if score.objective_value < best_score.objective_value:
                best_score = score

    runtime_seconds = perf_counter() - start
    return HestonCalibrationResult(
        parameters=best_score.parameters,
        objective_value=best_score.objective_value,
        rmse_price=best_score.rmse_price,
        rmse_implied_volatility=best_score.rmse_implied_volatility,
        max_abs_price_error=best_score.max_abs_price_error,
        num_quotes=len(quotes),
        evaluations=evaluations,
        runtime_seconds=runtime_seconds,
        fitted_points=best_score.fitted_points,
    )


def format_calibration_summary(result: HestonCalibrationResult) -> str:
    rmse_iv = "-" if result.rmse_implied_volatility is None else f"{result.rmse_implied_volatility:.5f}"
    params = result.parameters
    lines = [
        f"Quotes             : {result.num_quotes}",
        f"Objective          : {result.objective_value:.6f}",
        f"RMSE price         : {result.rmse_price:.5f}",
        f"RMSE implied vol   : {rmse_iv}",
        f"Max abs px error   : {result.max_abs_price_error:.5f}",
        f"v0                 : {params.initial_variance:.5f}",
        f"theta              : {params.long_run_variance:.5f}",
        f"kappa              : {params.mean_reversion:.5f}",
        f"xi                 : {params.vol_of_vol:.5f}",
        f"rho                : {params.correlation:.5f}",
        f"Evaluations        : {result.evaluations}",
        f"Runtime            : {result.runtime_seconds:.2f} s",
    ]
    return "\n".join(lines)


def format_calibration_table(
    points: tuple[CalibrationPoint, ...] | list[CalibrationPoint],
    *,
    max_rows: int | None = None,
) -> str:
    rows = list(points)
    if max_rows is not None:
        rows = rows[:max_rows]

    headers = (
        ("expiry", 12),
        ("strike", 9),
        ("market", 10),
        ("model", 10),
        ("abs err", 10),
        ("mkt iv", 10),
        ("mdl iv", 10),
    )
    lines = [" ".join(label.ljust(width) for label, width in headers)]
    lines.append(" ".join("-" * width for _, width in headers))
    for point in rows:
        market_iv = "-" if point.market_implied_volatility is None else f"{point.market_implied_volatility:.4f}"
        model_iv = "-" if point.model_implied_volatility is None else f"{point.model_implied_volatility:.4f}"
        values = (
            point.expiration,
            f"{point.strike:.2f}",
            f"{point.market_price:.4f}",
            f"{point.model_price:.4f}",
            f"{point.absolute_price_error:.4f}",
            market_iv,
            model_iv,
        )
        lines.append(" ".join(value.ljust(width) for value, (_, width) in zip(values, headers)))
    return "\n".join(lines)


def write_calibration_csv(
    result: HestonCalibrationResult,
    destination: str | Path,
) -> Path:
    output_path = Path(destination)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "expiration",
                "strike",
                "maturity",
                "market_price",
                "model_price",
                "absolute_price_error",
                "market_implied_volatility",
                "model_implied_volatility",
                "absolute_iv_error",
            ]
        )
        for point in result.fitted_points:
            writer.writerow(
                [
                    point.expiration,
                    point.strike,
                    point.maturity,
                    point.market_price,
                    point.model_price,
                    point.absolute_price_error,
                    point.market_implied_volatility,
                    point.model_implied_volatility,
                    point.absolute_iv_error,
                ]
            )
    return output_path


def _compute_maturity(snapshot_date: str, expiration: str) -> float:
    start = _parse_date(snapshot_date)
    end = _parse_date(expiration)
    days = (end - start).days
    if days <= 0:
        raise ValueError("expiration must be after snapshot_date.")
    return days / 365.0


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _sample_parameters(
    rng: np.random.Generator,
    bounds: dict[str, tuple[float, float]],
) -> HestonParameters:
    return HestonParameters(
        initial_variance=_sample_log_uniform(rng, *bounds["initial_variance"]),
        long_run_variance=_sample_log_uniform(rng, *bounds["long_run_variance"]),
        mean_reversion=_sample_log_uniform(rng, *bounds["mean_reversion"]),
        vol_of_vol=_sample_log_uniform(rng, *bounds["vol_of_vol"]),
        correlation=float(rng.uniform(*bounds["correlation"])),
    )


def _sample_log_uniform(
    rng: np.random.Generator,
    lower: float,
    upper: float,
) -> float:
    return float(math.exp(rng.uniform(math.log(lower), math.log(upper))))


def _coordinate_candidates(
    parameters: HestonParameters,
    scale: float,
    bounds: dict[str, tuple[float, float]],
) -> tuple[HestonParameters, ...]:
    candidates: list[HestonParameters] = []
    for attribute, (lower, upper) in bounds.items():
        width = upper - lower
        base_value = getattr(parameters, attribute)
        if attribute == "correlation":
            shifts = (-0.5 * scale * width, 0.5 * scale * width)
            for shift in shifts:
                candidate = _replace_parameter(
                    parameters,
                    attribute,
                    min(max(base_value + shift, lower), upper),
                )
                candidates.append(candidate)
            continue

        log_width = math.log(upper / lower)
        shifts = (-0.5 * scale * log_width, 0.5 * scale * log_width)
        for shift in shifts:
            bumped = math.exp(math.log(base_value) + shift)
            candidate = _replace_parameter(
                parameters,
                attribute,
                min(max(bumped, lower), upper),
            )
            candidates.append(candidate)
    return tuple(candidates)


def _replace_parameter(
    parameters: HestonParameters,
    attribute: str,
    value: float,
) -> HestonParameters:
    return HestonParameters(
        initial_variance=value if attribute == "initial_variance" else parameters.initial_variance,
        long_run_variance=value if attribute == "long_run_variance" else parameters.long_run_variance,
        mean_reversion=value if attribute == "mean_reversion" else parameters.mean_reversion,
        vol_of_vol=value if attribute == "vol_of_vol" else parameters.vol_of_vol,
        correlation=value if attribute == "correlation" else parameters.correlation,
    )


def _perturb_parameters(
    parameters: HestonParameters,
    rng: np.random.Generator,
    scale: float,
    bounds: dict[str, tuple[float, float]],
) -> HestonParameters:
    return HestonParameters(
        initial_variance=_perturb_positive(
            parameters.initial_variance,
            rng,
            scale,
            *bounds["initial_variance"],
        ),
        long_run_variance=_perturb_positive(
            parameters.long_run_variance,
            rng,
            scale,
            *bounds["long_run_variance"],
        ),
        mean_reversion=_perturb_positive(
            parameters.mean_reversion,
            rng,
            scale,
            *bounds["mean_reversion"],
        ),
        vol_of_vol=_perturb_positive(
            parameters.vol_of_vol,
            rng,
            scale,
            *bounds["vol_of_vol"],
        ),
        correlation=float(
            np.clip(
                parameters.correlation
                + rng.normal(0.0, scale * (bounds["correlation"][1] - bounds["correlation"][0]) * 0.35),
                bounds["correlation"][0],
                bounds["correlation"][1],
            )
        ),
    )


def _perturb_positive(
    value: float,
    rng: np.random.Generator,
    scale: float,
    lower: float,
    upper: float,
) -> float:
    span = math.log(upper / lower)
    perturbed = math.exp(math.log(value) + rng.normal(0.0, scale * span * 0.35))
    return float(np.clip(perturbed, lower, upper))


def _clip_parameters(
    parameters: HestonParameters,
    bounds: dict[str, tuple[float, float]],
) -> HestonParameters:
    return HestonParameters(
        initial_variance=float(np.clip(parameters.initial_variance, *bounds["initial_variance"])),
        long_run_variance=float(np.clip(parameters.long_run_variance, *bounds["long_run_variance"])),
        mean_reversion=float(np.clip(parameters.mean_reversion, *bounds["mean_reversion"])),
        vol_of_vol=float(np.clip(parameters.vol_of_vol, *bounds["vol_of_vol"])),
        correlation=float(np.clip(parameters.correlation, *bounds["correlation"])),
    )


def _evaluate_candidate(
    parameters: HestonParameters,
    quotes: list[MarketOptionQuote],
    *,
    integration_limit: float,
    integration_points: int,
) -> _CalibrationScore:
    fitted_points: list[CalibrationPoint] = []
    price_errors: list[float] = []
    iv_errors: list[float] = []

    for quote in quotes:
        spec = HestonSpec(
            spot=quote.spot,
            strike=quote.strike,
            rate=quote.rate,
            maturity=quote.maturity,
            initial_variance=parameters.initial_variance,
            long_run_variance=parameters.long_run_variance,
            mean_reversion=parameters.mean_reversion,
            vol_of_vol=parameters.vol_of_vol,
            correlation=parameters.correlation,
            option_type=quote.option_type,
            dividend_yield=quote.dividend_yield,
        )
        model_price = price_heston_option_cf(
            spec,
            integration_limit=integration_limit,
            integration_points=integration_points,
        )
        model_iv = implied_volatility_from_price(
            model_price,
            spot=quote.spot,
            strike=quote.strike,
            rate=quote.rate,
            maturity=quote.maturity,
            option_type=quote.option_type,
            dividend_yield=quote.dividend_yield,
        )
        absolute_price_error = abs(model_price - quote.market_price)
        absolute_iv_error = (
            None
            if quote.market_implied_volatility is None or model_iv is None
            else abs(model_iv - quote.market_implied_volatility)
        )
        fitted_points.append(
            CalibrationPoint(
                expiration=quote.expiration,
                strike=quote.strike,
                maturity=quote.maturity,
                market_price=quote.market_price,
                model_price=model_price,
                absolute_price_error=absolute_price_error,
                market_implied_volatility=quote.market_implied_volatility,
                model_implied_volatility=model_iv,
                absolute_iv_error=absolute_iv_error,
            )
        )
        price_errors.append(model_price - quote.market_price)
        if absolute_iv_error is not None:
            iv_errors.append(model_iv - quote.market_implied_volatility)

    rmse_price = math.sqrt(float(np.mean(np.square(price_errors))))
    rmse_iv = math.sqrt(float(np.mean(np.square(iv_errors)))) if iv_errors else None
    average_price = max(float(np.mean([quote.market_price for quote in quotes])), 1e-8)
    objective_value = rmse_price / average_price
    if rmse_iv is not None:
        objective_value += 0.5 * rmse_iv

    # Softly discourage candidates that violate the Feller condition badly.
    feller_gap = max(
        0.0,
        parameters.vol_of_vol * parameters.vol_of_vol
        - 2.0 * parameters.mean_reversion * parameters.long_run_variance,
    )
    objective_value += 0.02 * feller_gap

    return _CalibrationScore(
        parameters=parameters,
        objective_value=objective_value,
        rmse_price=rmse_price,
        rmse_implied_volatility=rmse_iv,
        max_abs_price_error=max(point.absolute_price_error for point in fitted_points),
        fitted_points=tuple(fitted_points),
    )
