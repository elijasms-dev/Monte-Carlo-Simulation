from __future__ import annotations

import argparse

from .pricing import black_scholes_price, price_european_option_mc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Price a European option with Monte Carlo simulation."
    )
    parser.add_argument("--spot", type=float, default=100.0, help="Current spot price.")
    parser.add_argument("--strike", type=float, default=100.0, help="Option strike.")
    parser.add_argument(
        "--rate", type=float, default=0.05, help="Continuously compounded risk-free rate."
    )
    parser.add_argument(
        "--vol", type=float, default=0.2, help="Annualized volatility."
    )
    parser.add_argument(
        "--maturity", type=float, default=1.0, help="Time to maturity in years."
    )
    parser.add_argument(
        "--paths", type=int, default=100_000, help="Number of Monte Carlo paths."
    )
    parser.add_argument(
        "--option-type",
        choices=("call", "put"),
        default="call",
        help="European option type.",
    )
    parser.add_argument(
        "--seed", type=int, default=7, help="Seed used for reproducible draws."
    )
    parser.add_argument(
        "--no-antithetic",
        action="store_true",
        help="Disable antithetic variates.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = price_european_option_mc(
        spot=args.spot,
        strike=args.strike,
        rate=args.rate,
        volatility=args.vol,
        maturity=args.maturity,
        num_paths=args.paths,
        option_type=args.option_type,
        seed=args.seed,
        antithetic=not args.no_antithetic,
    )
    benchmark = black_scholes_price(
        spot=args.spot,
        strike=args.strike,
        rate=args.rate,
        volatility=args.vol,
        maturity=args.maturity,
        option_type=args.option_type,
    )

    print(f"Monte Carlo price : {result.price:.4f}")
    print(f"Std. error        : {result.standard_error:.4f}")
    print(
        f"95% CI            : [{result.confidence_interval[0]:.4f}, "
        f"{result.confidence_interval[1]:.4f}]"
    )
    print(f"Black-Scholes     : {benchmark:.4f}")
    print(f"Absolute error    : {abs(result.price - benchmark):.4f}")

