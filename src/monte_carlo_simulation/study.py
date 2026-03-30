from __future__ import annotations

import csv
from pathlib import Path

from .pricing import OptionSpec, StudyResult, build_method_config, price_option_mc


DEFAULT_METHODS = (
    "naive",
    "antithetic",
    "antithetic-control-variate",
    "halton",
    "halton-control-variate",
)


def run_convergence_study(
    spec: OptionSpec,
    path_counts: list[int],
    *,
    time_steps: int = 252,
    methods: tuple[str, ...] = DEFAULT_METHODS,
    seed: int | None = None,
) -> list[StudyResult]:
    results: list[StudyResult] = []
    warmup_paths = min(path_counts)
    warmup_config = build_method_config(
        methods[0],
        num_paths=max(256, min(1_024, warmup_paths)),
        time_steps=time_steps,
        seed=seed,
    )
    price_option_mc(spec=spec, config=warmup_config, include_greeks=False)

    for num_paths in path_counts:
        baseline_metric: float | None = None
        for method in methods:
            config = build_method_config(
                method,
                num_paths=num_paths,
                time_steps=time_steps,
                seed=seed,
            )
            result = price_option_mc(spec=spec, config=config, include_greeks=False)
            variance_time = max(result.standard_error * result.standard_error * result.runtime_seconds, 1e-12)
            if baseline_metric is None:
                baseline_metric = variance_time
                efficiency_gain = 1.0
            else:
                efficiency_gain = baseline_metric / variance_time

            results.append(
                StudyResult(
                    num_paths=num_paths,
                    method=result.method,
                    price=result.price,
                    standard_error=result.standard_error,
                    confidence_interval=result.confidence_interval,
                    benchmark_price=result.benchmark_price,
                    absolute_error=result.absolute_error,
                    runtime_seconds=result.runtime_seconds,
                    efficiency_gain=efficiency_gain,
                )
            )

    return results


def format_study_table(results: list[StudyResult]) -> str:
    headers = (
        ("paths", 8),
        ("method", 30),
        ("price", 11),
        ("stderr", 11),
        ("abs err", 11),
        ("runtime ms", 12),
        ("eff gain", 10),
    )
    lines = [" ".join(label.ljust(width) for label, width in headers)]
    lines.append(" ".join("-" * width for _, width in headers))

    for row in results:
        absolute_error = "-" if row.absolute_error is None else f"{row.absolute_error:.5f}"
        efficiency_gain = "-" if row.efficiency_gain is None else f"{row.efficiency_gain:.2f}x"
        values = (
            str(row.num_paths),
            row.method,
            f"{row.price:.5f}",
            f"{row.standard_error:.5f}",
            absolute_error,
            f"{row.runtime_seconds * 1000:.2f}",
            efficiency_gain,
        )
        lines.append(
            " ".join(value.ljust(width) for value, (_, width) in zip(values, headers))
        )

    return "\n".join(lines)


def write_study_csv(results: list[StudyResult], destination: str) -> Path:
    output_path = Path(destination)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "num_paths",
                "method",
                "price",
                "standard_error",
                "ci_low",
                "ci_high",
                "benchmark_price",
                "absolute_error",
                "runtime_seconds",
                "efficiency_gain",
            ]
        )
        for row in results:
            writer.writerow(
                [
                    row.num_paths,
                    row.method,
                    row.price,
                    row.standard_error,
                    row.confidence_interval[0],
                    row.confidence_interval[1],
                    row.benchmark_price,
                    row.absolute_error,
                    row.runtime_seconds,
                    row.efficiency_gain,
                ]
            )
    return output_path
