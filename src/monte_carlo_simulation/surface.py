from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from pathlib import Path

from .pricing import OptionSpec, SimulationConfig, price_option_mc


@dataclass(frozen=True)
class SurfacePoint:
    spot: float
    volatility: float
    price: float
    standard_error: float
    benchmark_price: float | None
    absolute_error: float | None
    delta: float | None
    gamma: float | None
    vega: float | None


def run_sensitivity_surface(
    spec: OptionSpec,
    config: SimulationConfig,
    *,
    spots: list[float],
    volatilities: list[float],
    include_greeks: bool = True,
) -> list[SurfacePoint]:
    points: list[SurfacePoint] = []
    for spot in spots:
        for volatility in volatilities:
            scenario = replace(spec, spot=spot, volatility=volatility)
            result = price_option_mc(
                spec=scenario,
                config=config,
                include_greeks=include_greeks and scenario.payoff == "european",
            )
            greeks = result.greeks
            points.append(
                SurfacePoint(
                    spot=spot,
                    volatility=volatility,
                    price=result.price,
                    standard_error=result.standard_error,
                    benchmark_price=result.benchmark_price,
                    absolute_error=result.absolute_error,
                    delta=None if greeks is None else greeks.delta,
                    gamma=None if greeks is None else greeks.gamma,
                    vega=None if greeks is None else greeks.vega,
                )
            )
    return points


def format_surface_table(points: list[SurfacePoint]) -> str:
    headers = (
        ("spot", 8),
        ("vol", 8),
        ("price", 11),
        ("stderr", 11),
        ("delta", 11),
        ("vega", 11),
        ("abs err", 11),
    )
    lines = [" ".join(label.ljust(width) for label, width in headers)]
    lines.append(" ".join("-" * width for _, width in headers))
    for point in points:
        values = (
            f"{point.spot:.2f}",
            f"{point.volatility:.2f}",
            f"{point.price:.5f}",
            f"{point.standard_error:.5f}",
            "-" if point.delta is None else f"{point.delta:.5f}",
            "-" if point.vega is None else f"{point.vega:.5f}",
            "-" if point.absolute_error is None else f"{point.absolute_error:.5f}",
        )
        lines.append(
            " ".join(value.ljust(width) for value, (_, width) in zip(values, headers))
        )
    return "\n".join(lines)


def write_surface_csv(points: list[SurfacePoint], destination: str) -> Path:
    output_path = Path(destination)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "spot",
                "volatility",
                "price",
                "standard_error",
                "benchmark_price",
                "absolute_error",
                "delta",
                "gamma",
                "vega",
            ]
        )
        for point in points:
            writer.writerow(
                [
                    point.spot,
                    point.volatility,
                    point.price,
                    point.standard_error,
                    point.benchmark_price,
                    point.absolute_error,
                    point.delta,
                    point.gamma,
                    point.vega,
                ]
            )
    return output_path
