from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from monte_carlo_simulation.pricing import black_scholes_price, price_european_option_mc


class PricingTests(unittest.TestCase):
    def test_call_price_tracks_black_scholes(self) -> None:
        result = price_european_option_mc(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.2,
            maturity=1.0,
            num_paths=120_000,
            option_type="call",
            seed=7,
        )
        benchmark = black_scholes_price(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.2,
            maturity=1.0,
            option_type="call",
        )

        self.assertAlmostEqual(result.price, benchmark, delta=0.35)
        self.assertLess(abs(result.price - benchmark), 2 * result.standard_error + 0.05)

    def test_put_price_tracks_black_scholes(self) -> None:
        result = price_european_option_mc(
            spot=100.0,
            strike=95.0,
            rate=0.03,
            volatility=0.25,
            maturity=1.5,
            num_paths=120_000,
            option_type="put",
            seed=11,
        )
        benchmark = black_scholes_price(
            spot=100.0,
            strike=95.0,
            rate=0.03,
            volatility=0.25,
            maturity=1.5,
            option_type="put",
        )

        self.assertAlmostEqual(result.price, benchmark, delta=0.4)

    def test_zero_volatility_is_deterministic(self) -> None:
        result = price_european_option_mc(
            spot=105.0,
            strike=100.0,
            rate=0.01,
            volatility=0.0,
            maturity=2.0,
            num_paths=10_000,
            option_type="call",
            seed=1,
        )

        self.assertEqual(result.standard_error, 0.0)
        self.assertEqual(result.price, black_scholes_price(105.0, 100.0, 0.01, 0.0, 2.0, "call"))


if __name__ == "__main__":
    unittest.main()
