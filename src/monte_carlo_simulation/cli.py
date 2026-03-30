from __future__ import annotations

import argparse
import sys

from .american import price_american_option_lsm
from .basket import (
    BASKET_ARITHMETIC,
    BASKET_GEOMETRIC,
    BasketOptionSpec,
    build_equicorrelation_matrix,
    geometric_basket_price,
    price_basket_option_mc,
)
from .calibration import (
    bundled_market_data_path,
    calibrate_heston_parameters,
    format_calibration_summary,
    format_calibration_table,
    load_market_quotes_csv,
    write_calibration_csv,
)
from .heston import (
    HestonSpec,
    format_heston_smile_table,
    price_heston_option_cf,
    price_heston_option_mc,
    run_heston_smile,
    run_heston_smile_cf,
    write_heston_smile_csv,
)
from .pricing import (
    ASIAN_ARITHMETIC,
    EUROPEAN,
    UP_AND_OUT,
    MonteCarloResult,
    OptionSpec,
    SimulationConfig,
    black_scholes_delta,
    black_scholes_gamma,
    black_scholes_vega,
    price_option_mc,
)
from .report import build_research_report
from .study import DEFAULT_METHODS, format_study_table, run_convergence_study, write_study_csv
from .surface import format_surface_table, run_sensitivity_surface, write_surface_csv


def _add_option_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--spot", type=float, default=100.0, help="Current spot price.")
    parser.add_argument("--strike", type=float, default=100.0, help="Option strike.")
    parser.add_argument(
        "--rate",
        type=float,
        default=0.05,
        help="Continuously compounded risk-free rate.",
    )
    parser.add_argument("--vol", type=float, default=0.2, help="Annualized volatility.")
    parser.add_argument(
        "--maturity", type=float, default=1.0, help="Time to maturity in years."
    )
    parser.add_argument(
        "--option-type",
        choices=("call", "put"),
        default="call",
        help="Option direction.",
    )


def _add_payoff_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--payoff",
        choices=(EUROPEAN, ASIAN_ARITHMETIC, UP_AND_OUT),
        default=EUROPEAN,
        help="Payoff family to price.",
    )
    parser.add_argument(
        "--barrier",
        type=float,
        default=None,
        help="Barrier level required for up-and-out options.",
    )


def _add_simulation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--paths", type=int, default=100_000, help="Number of simulation paths.")
    parser.add_argument(
        "--steps", type=int, default=252, help="Monitoring dates used in path simulation."
    )
    parser.add_argument("--seed", type=int, default=7, help="Seed used for reproducible draws.")
    parser.add_argument(
        "--no-antithetic",
        action="store_true",
        help="Disable antithetic variates.",
    )
    parser.add_argument(
        "--control-variate",
        action="store_true",
        help="Use discounted terminal spot as a control variate.",
    )
    parser.add_argument(
        "--sampling",
        choices=("pseudo", "halton"),
        default="pseudo",
        help="Sampling scheme used to generate Gaussian shocks.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Quant-facing Monte Carlo pricing and experiment toolkit."
    )
    subparsers = parser.add_subparsers(dest="command")

    price_parser = subparsers.add_parser("price", help="Price a single option and report diagnostics.")
    _add_option_arguments(price_parser)
    _add_payoff_arguments(price_parser)
    _add_simulation_arguments(price_parser)
    price_parser.add_argument(
        "--no-greeks",
        action="store_true",
        help="Skip finite-difference Greeks for European options.",
    )

    study_parser = subparsers.add_parser(
        "study",
        help="Run a convergence study across variance-reduction methods.",
    )
    _add_option_arguments(study_parser)
    _add_payoff_arguments(study_parser)
    study_parser.add_argument(
        "--path-counts",
        type=int,
        nargs="+",
        default=[5_000, 20_000, 50_000, 100_000],
        help="Path counts to evaluate in the study.",
    )
    study_parser.add_argument("--steps", type=int, default=252, help="Monitoring dates used in path simulation.")
    study_parser.add_argument("--seed", type=int, default=7, help="Seed used for reproducible draws.")
    study_parser.add_argument(
        "--methods",
        nargs="+",
        choices=DEFAULT_METHODS,
        default=list(DEFAULT_METHODS),
        help="Variance-reduction methods to compare.",
    )
    study_parser.add_argument("--csv", type=str, default=None, help="Optional CSV output path.")

    surface_parser = subparsers.add_parser(
        "surface",
        help="Generate a price and Greek surface across spot/volatility scenarios.",
    )
    _add_option_arguments(surface_parser)
    _add_payoff_arguments(surface_parser)
    _add_simulation_arguments(surface_parser)
    surface_parser.add_argument(
        "--spot-grid",
        type=float,
        nargs="+",
        default=[80.0, 90.0, 100.0, 110.0, 120.0],
        help="Spot values to evaluate.",
    )
    surface_parser.add_argument(
        "--vol-grid",
        type=float,
        nargs="+",
        default=[0.15, 0.20, 0.25, 0.30],
        help="Volatility values to evaluate.",
    )
    surface_parser.add_argument(
        "--no-greeks",
        action="store_true",
        help="Skip Greek estimation while building the surface.",
    )
    surface_parser.add_argument("--csv", type=str, default=None, help="Optional CSV output path.")

    american_parser = subparsers.add_parser(
        "american",
        help="Price an American vanilla option with Longstaff-Schwartz.",
    )
    _add_option_arguments(american_parser)
    _add_simulation_arguments(american_parser)
    american_parser.add_argument(
        "--basis-order",
        type=int,
        default=2,
        help="Polynomial basis order used in the continuation regression.",
    )

    basket_parser = subparsers.add_parser(
        "basket",
        help="Price a correlated basket option with Monte Carlo.",
    )
    basket_parser.add_argument("--spots", type=float, nargs="+", required=True, help="Spot values for each asset.")
    basket_parser.add_argument(
        "--vols",
        type=float,
        nargs="+",
        required=True,
        help="Volatilities for each asset in the basket.",
    )
    basket_parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=None,
        help="Weights for each asset. Defaults to equal weights.",
    )
    basket_parser.add_argument("--strike", type=float, default=100.0, help="Basket option strike.")
    basket_parser.add_argument("--rate", type=float, default=0.05, help="Risk-free rate.")
    basket_parser.add_argument("--maturity", type=float, default=1.0, help="Time to maturity in years.")
    basket_parser.add_argument(
        "--option-type",
        choices=("call", "put"),
        default="call",
        help="Option direction.",
    )
    basket_parser.add_argument(
        "--basket-payoff",
        choices=(BASKET_ARITHMETIC, BASKET_GEOMETRIC),
        default=BASKET_ARITHMETIC,
        help="Basket payoff family.",
    )
    basket_parser.add_argument(
        "--rho",
        type=float,
        default=0.35,
        help="Equicorrelation used when --corr is not provided.",
    )
    basket_parser.add_argument(
        "--corr",
        type=float,
        nargs="*",
        default=None,
        help="Flattened correlation matrix values in row-major order.",
    )
    basket_parser.add_argument("--paths", type=int, default=100_000, help="Number of simulation paths.")
    basket_parser.add_argument("--seed", type=int, default=7, help="Seed used for reproducible draws.")
    basket_parser.add_argument(
        "--no-antithetic",
        action="store_true",
        help="Disable antithetic variates.",
    )
    basket_parser.add_argument(
        "--sampling",
        choices=("pseudo", "halton"),
        default="pseudo",
        help="Sampling scheme used to generate correlated shocks.",
    )
    basket_parser.add_argument(
        "--no-geometric-cv",
        action="store_true",
        help="Disable the geometric-basket control variate for arithmetic baskets.",
    )

    heston_parser = subparsers.add_parser(
        "heston",
        help="Price a European option under Heston stochastic volatility.",
    )
    _add_option_arguments(heston_parser)
    _add_simulation_arguments(heston_parser)
    heston_parser.add_argument(
        "--v0",
        type=float,
        default=0.04,
        help="Initial variance level.",
    )
    heston_parser.add_argument(
        "--theta",
        type=float,
        default=0.04,
        help="Long-run variance level.",
    )
    heston_parser.add_argument(
        "--kappa",
        type=float,
        default=2.0,
        help="Mean-reversion speed of the variance process.",
    )
    heston_parser.add_argument(
        "--vol-of-vol",
        type=float,
        default=0.5,
        help="Volatility of variance.",
    )
    heston_parser.add_argument(
        "--rho",
        type=float,
        default=-0.7,
        help="Correlation between asset and variance shocks.",
    )
    heston_parser.add_argument(
        "--dividend-yield",
        type=float,
        default=0.0,
        help="Continuous dividend yield used in the Heston drift.",
    )
    heston_parser.add_argument(
        "--cf",
        action="store_true",
        help="Use the semi-closed-form Heston characteristic-function pricer.",
    )

    smile_parser = subparsers.add_parser(
        "smile",
        help="Generate a Heston implied-volatility smile across strikes.",
    )
    _add_option_arguments(smile_parser)
    _add_simulation_arguments(smile_parser)
    smile_parser.add_argument(
        "--v0",
        type=float,
        default=0.04,
        help="Initial variance level.",
    )
    smile_parser.add_argument(
        "--theta",
        type=float,
        default=0.04,
        help="Long-run variance level.",
    )
    smile_parser.add_argument(
        "--kappa",
        type=float,
        default=2.0,
        help="Mean-reversion speed of the variance process.",
    )
    smile_parser.add_argument(
        "--vol-of-vol",
        type=float,
        default=0.5,
        help="Volatility of variance.",
    )
    smile_parser.add_argument(
        "--rho",
        type=float,
        default=-0.7,
        help="Correlation between asset and variance shocks.",
    )
    smile_parser.add_argument(
        "--dividend-yield",
        type=float,
        default=0.0,
        help="Continuous dividend yield used in the Heston drift.",
    )
    smile_parser.add_argument(
        "--strikes",
        type=float,
        nargs="+",
        default=[80.0, 90.0, 100.0, 110.0, 120.0],
        help="Strike grid used for the smile study.",
    )
    smile_parser.add_argument(
        "--cf",
        action="store_true",
        help="Use the semi-closed-form Heston characteristic-function pricer.",
    )
    smile_parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional CSV output path.",
    )

    calibrate_parser = subparsers.add_parser(
        "calibrate",
        help="Calibrate Heston parameters to a market option snapshot.",
    )
    calibrate_parser.add_argument(
        "--market-data",
        type=str,
        default=str(bundled_market_data_path()),
        help="CSV file containing market option quotes.",
    )
    calibrate_parser.add_argument(
        "--global-samples",
        type=int,
        default=48,
        help="Number of broad random candidates evaluated before local refinement.",
    )
    calibrate_parser.add_argument(
        "--rounds",
        type=int,
        default=6,
        help="Number of local-search refinement rounds.",
    )
    calibrate_parser.add_argument(
        "--local-samples",
        type=int,
        default=24,
        help="Number of local candidates evaluated per refinement round.",
    )
    calibrate_parser.add_argument(
        "--integration-points",
        type=int,
        default=256,
        help="Quadrature points used in the Heston characteristic-function pricer.",
    )
    calibrate_parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional CSV output path for fitted-vs-market rows.",
    )

    report_parser = subparsers.add_parser(
        "report",
        help="Generate a markdown research report from the current toolkit.",
    )
    report_parser.add_argument(
        "--output",
        type=str,
        default="outputs/research_report.md",
        help="Markdown output path for the generated report.",
    )
    report_parser.add_argument(
        "--calibration-data",
        type=str,
        default=str(bundled_market_data_path()),
        help="Market snapshot used for the calibration section of the report.",
    )
    report_parser.add_argument("--seed", type=int, default=7, help="Seed used for reproducible draws.")
    return parser


def _normalize_argv(argv: list[str]) -> list[str]:
    if not argv:
        return ["price"]
    if argv[0] in {"price", "study", "surface", "american", "basket", "heston", "smile", "calibrate", "report", "-h", "--help"}:
        return argv
    return ["price", *argv]


def _build_option_spec(args: argparse.Namespace) -> OptionSpec:
    return OptionSpec(
        spot=args.spot,
        strike=args.strike,
        rate=args.rate,
        volatility=args.vol,
        maturity=args.maturity,
        option_type=args.option_type,
        payoff=getattr(args, "payoff", EUROPEAN),
        barrier=getattr(args, "barrier", None),
    )


def _build_config(args: argparse.Namespace) -> SimulationConfig:
    return SimulationConfig(
        num_paths=args.paths,
        time_steps=getattr(args, "steps", 1),
        seed=args.seed,
        antithetic=not args.no_antithetic,
        control_variate=getattr(args, "control_variate", False),
        sampling=getattr(args, "sampling", "pseudo"),
    )


def _build_price_lines(spec: OptionSpec, result: MonteCarloResult) -> list[str]:
    lines = [
        f"Payoff            : {spec.payoff} {spec.option_type}",
        f"Method            : {result.method}",
        f"Paths x steps     : {result.num_paths} x {result.time_steps}",
        f"Monte Carlo price : {result.price:.5f}",
        f"Std. error        : {result.standard_error:.5f}",
        f"95% CI            : [{result.confidence_interval[0]:.5f}, {result.confidence_interval[1]:.5f}]",
        f"Runtime           : {result.runtime_seconds * 1000:.2f} ms",
    ]
    if result.benchmark_price is not None:
        lines.append(f"Benchmark         : {result.benchmark_price:.5f}")
        lines.append(f"Absolute error    : {result.absolute_error:.5f}")
    if result.greeks is not None:
        lines.append(f"Delta (MC)        : {result.greeks.delta:.5f}")
        lines.append(f"Gamma (MC)        : {result.greeks.gamma:.5f}")
        lines.append(f"Vega (MC)         : {result.greeks.vega:.5f}")
        lines.append(
            f"Delta (BS)        : {black_scholes_delta(spec.spot, spec.strike, spec.rate, spec.volatility, spec.maturity, spec.option_type):.5f}"
        )
        lines.append(
            f"Gamma (BS)        : {black_scholes_gamma(spec.spot, spec.strike, spec.rate, spec.volatility, spec.maturity):.5f}"
        )
        lines.append(
            f"Vega (BS)         : {black_scholes_vega(spec.spot, spec.strike, spec.rate, spec.volatility, spec.maturity):.5f}"
        )
    if spec.payoff == UP_AND_OUT:
        lines.append(f"Barrier           : {spec.barrier:.5f}")
    return lines


def _build_weights(raw_weights: list[float] | None, dimension: int) -> tuple[float, ...]:
    if raw_weights is None:
        equal = 1.0 / dimension
        return tuple(equal for _ in range(dimension))
    if len(raw_weights) != dimension:
        raise ValueError("weights must match the number of basket assets.")
    total = sum(raw_weights)
    if total == 0:
        raise ValueError("weights must not sum to zero.")
    return tuple(weight / total for weight in raw_weights)


def _build_correlation(args: argparse.Namespace, dimension: int) -> tuple[tuple[float, ...], ...]:
    if args.corr is None:
        return build_equicorrelation_matrix(dimension, args.rho)
    if len(args.corr) != dimension * dimension:
        raise ValueError("corr must contain dimension^2 entries.")
    rows: list[tuple[float, ...]] = []
    for row in range(dimension):
        start = row * dimension
        rows.append(tuple(args.corr[start : start + dimension]))
    return tuple(rows)


def _build_heston_spec(args: argparse.Namespace) -> HestonSpec:
    return HestonSpec(
        spot=args.spot,
        strike=args.strike,
        rate=args.rate,
        maturity=args.maturity,
        initial_variance=args.v0,
        long_run_variance=args.theta,
        mean_reversion=args.kappa,
        vol_of_vol=args.vol_of_vol,
        correlation=args.rho,
        option_type=args.option_type,
        dividend_yield=args.dividend_yield,
    )


def main(argv: list[str] | None = None) -> None:
    raw_argv = sys.argv[1:] if argv is None else argv
    parser = build_parser()
    args = parser.parse_args(_normalize_argv(raw_argv))

    try:
        if args.command == "price":
            spec = _build_option_spec(args)
            config = _build_config(args)
            include_greeks = spec.payoff == EUROPEAN and not args.no_greeks
            result = price_option_mc(spec=spec, config=config, include_greeks=include_greeks)
            print("\n".join(_build_price_lines(spec, result)))
            return

        if args.command == "study":
            spec = _build_option_spec(args)
            results = run_convergence_study(
                spec=spec,
                path_counts=list(args.path_counts),
                time_steps=args.steps,
                methods=tuple(args.methods),
                seed=args.seed,
            )
            print(format_study_table(results))
            if args.csv:
                output_path = write_study_csv(results, args.csv)
                print(f"\nCSV written to    : {output_path}")
            return

        if args.command == "surface":
            spec = _build_option_spec(args)
            config = _build_config(args)
            points = run_sensitivity_surface(
                spec=spec,
                config=config,
                spots=list(args.spot_grid),
                volatilities=list(args.vol_grid),
                include_greeks=not args.no_greeks,
            )
            print(format_surface_table(points))
            if args.csv:
                output_path = write_surface_csv(points, args.csv)
                print(f"\nCSV written to    : {output_path}")
            return

        if args.command == "american":
            spec = OptionSpec(
                spot=args.spot,
                strike=args.strike,
                rate=args.rate,
                volatility=args.vol,
                maturity=args.maturity,
                option_type=args.option_type,
                payoff=EUROPEAN,
            )
            config = _build_config(args)
            result = price_american_option_lsm(spec=spec, config=config, basis_order=args.basis_order)
            lines = [
                f"Option            : American {spec.option_type}",
                f"Method            : {result.method}",
                f"Paths x steps     : {result.num_paths} x {result.time_steps}",
                f"LSM price         : {result.price:.5f}",
                f"Std. error        : {result.standard_error:.5f}",
                f"95% CI            : [{result.confidence_interval[0]:.5f}, {result.confidence_interval[1]:.5f}]",
                f"European ref      : {result.european_reference_price:.5f}",
                f"Exercise premium  : {result.premium_over_european:.5f}",
                f"Early exercise %  : {100.0 * result.early_exercise_ratio:.2f}",
                f"Runtime           : {result.runtime_seconds * 1000:.2f} ms",
            ]
            print("\n".join(lines))
            return

        if args.command == "basket":
            dimension = len(args.spots)
            if len(args.vols) != dimension:
                raise ValueError("spots and vols must have the same length.")
            spec = BasketOptionSpec(
                spots=tuple(args.spots),
                volatilities=tuple(args.vols),
                weights=_build_weights(args.weights, dimension),
                correlation_matrix=_build_correlation(args, dimension),
                strike=args.strike,
                rate=args.rate,
                maturity=args.maturity,
                option_type=args.option_type,
                payoff=args.basket_payoff,
            )
            config = SimulationConfig(
                num_paths=args.paths,
                time_steps=1,
                seed=args.seed,
                antithetic=not args.no_antithetic,
                control_variate=False,
                sampling=args.sampling,
            )
            result = price_basket_option_mc(
                spec=spec,
                config=config,
                use_geometric_control_variate=not args.no_geometric_cv,
            )
            lines = [
                f"Basket payoff     : {spec.payoff} {spec.option_type}",
                f"Dimension         : {spec.dimension}",
                f"Method            : {result.method}",
                f"Monte Carlo price : {result.price:.5f}",
                f"Std. error        : {result.standard_error:.5f}",
                f"95% CI            : [{result.confidence_interval[0]:.5f}, {result.confidence_interval[1]:.5f}]",
                f"Runtime           : {result.runtime_seconds * 1000:.2f} ms",
            ]
            if result.benchmark_price is not None:
                lines.append(f"Geometric ref     : {result.benchmark_price:.5f}")
                lines.append(f"Absolute error    : {result.absolute_error:.5f}")
            elif spec.payoff == BASKET_ARITHMETIC and not args.no_geometric_cv:
                geometric_ref = geometric_basket_price(
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
                lines.append(f"Geometric CV ref  : {geometric_ref:.5f}")
            print("\n".join(lines))
            return

        if args.command == "heston":
            spec = _build_heston_spec(args)
            if args.cf:
                price = price_heston_option_cf(spec)
                lines = [
                    f"Model             : Heston stochastic volatility",
                    f"Method            : Heston characteristic function",
                    f"Closed-form price : {price:.5f}",
                    f"Dividend yield    : {args.dividend_yield:.5f}",
                ]
            else:
                config = _build_config(args)
                result = price_heston_option_mc(spec=spec, config=config)
                implied_vol = "-" if result.implied_volatility is None else f"{result.implied_volatility:.5f}"
                lines = [
                    f"Model             : Heston stochastic volatility",
                    f"Method            : {result.method}",
                    f"Paths x steps     : {result.num_paths} x {result.time_steps}",
                    f"Monte Carlo price : {result.price:.5f}",
                    f"Std. error        : {result.standard_error:.5f}",
                    f"95% CI            : [{result.confidence_interval[0]:.5f}, {result.confidence_interval[1]:.5f}]",
                    f"Implied vol       : {implied_vol}",
                    f"Mean path var     : {result.mean_path_variance:.5f}",
                    f"Mean terminal var : {result.mean_terminal_variance:.5f}",
                    f"Dividend yield    : {args.dividend_yield:.5f}",
                    f"Runtime           : {result.runtime_seconds * 1000:.2f} ms",
                ]
            print("\n".join(lines))
            return

        if args.command == "smile":
            spec = _build_heston_spec(args)
            if args.cf:
                points = run_heston_smile_cf(spec=spec, strikes=list(args.strikes))
            else:
                config = _build_config(args)
                points = run_heston_smile(spec=spec, strikes=list(args.strikes), config=config)
            print(format_heston_smile_table(points))
            if args.csv:
                output_path = write_heston_smile_csv(points, args.csv)
                print(f"\nCSV written to    : {output_path}")
            return

        if args.command == "calibrate":
            quotes = load_market_quotes_csv(args.market_data)
            result = calibrate_heston_parameters(
                quotes,
                seed=7,
                global_samples=args.global_samples,
                search_rounds=args.rounds,
                local_samples=args.local_samples,
                integration_points=args.integration_points,
            )
            print(f"Dataset            : {args.market_data}")
            print(format_calibration_summary(result))
            print()
            print(format_calibration_table(result.fitted_points))
            if args.csv:
                output_path = write_calibration_csv(result, args.csv)
                print(f"\nCSV written to    : {output_path}")
            return

        if args.command == "report":
            output_path = build_research_report(
                args.output,
                seed=args.seed,
                calibration_data=args.calibration_data,
            )
            print(f"Research report   : {output_path}")
            return
    except ValueError as exc:
        parser.error(str(exc))
