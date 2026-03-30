from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from monte_carlo_simulation.american import price_american_option_lsm
from monte_carlo_simulation.basket import (
    BASKET_ARITHMETIC,
    BASKET_GEOMETRIC,
    BasketOptionSpec,
    build_equicorrelation_matrix,
    geometric_basket_price,
    price_basket_option_mc,
)
from monte_carlo_simulation.calibration import (
    HestonParameters,
    MarketOptionQuote,
    bundled_market_data_path,
    calibrate_heston_parameters,
    load_market_quotes_csv,
)
from monte_carlo_simulation.heston import (
    HestonSpec,
    implied_volatility_from_price,
    price_heston_option_cf,
    price_heston_option_mc,
    run_heston_smile_cf,
)
from monte_carlo_simulation.pricing import (
    ASIAN_ARITHMETIC,
    CALL,
    EUROPEAN,
    PUT,
    UP_AND_OUT,
    OptionSpec,
    SimulationConfig,
    black_scholes_delta,
    black_scholes_gamma,
    black_scholes_price,
    black_scholes_vega,
    price_option_mc,
)
from monte_carlo_simulation.report import build_research_report
from monte_carlo_simulation.study import run_convergence_study
from monte_carlo_simulation.surface import run_sensitivity_surface


class PricingTests(unittest.TestCase):
    def test_european_call_tracks_black_scholes_and_greeks(self) -> None:
        spec = OptionSpec(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.2,
            maturity=1.0,
            option_type=CALL,
            payoff=EUROPEAN,
        )
        config = SimulationConfig(
            num_paths=60_000,
            time_steps=1,
            seed=7,
            antithetic=True,
            control_variate=True,
        )

        result = price_option_mc(spec=spec, config=config, include_greeks=True)

        self.assertAlmostEqual(
            result.price,
            black_scholes_price(100.0, 100.0, 0.05, 0.2, 1.0, CALL),
            delta=0.08,
        )
        self.assertIsNotNone(result.greeks)
        self.assertAlmostEqual(
            result.greeks.delta,
            black_scholes_delta(100.0, 100.0, 0.05, 0.2, 1.0, CALL),
            delta=0.03,
        )
        self.assertAlmostEqual(
            result.greeks.gamma,
            black_scholes_gamma(100.0, 100.0, 0.05, 0.2, 1.0),
            delta=0.01,
        )
        self.assertAlmostEqual(
            result.greeks.vega,
            black_scholes_vega(100.0, 100.0, 0.05, 0.2, 1.0),
            delta=1.5,
        )

    def test_variance_reduction_improves_standard_error(self) -> None:
        spec = OptionSpec(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.2,
            maturity=1.0,
            payoff=EUROPEAN,
        )
        naive = price_option_mc(
            spec=spec,
            config=SimulationConfig(
                num_paths=8_000,
                time_steps=1,
                seed=11,
                antithetic=False,
                control_variate=False,
            ),
            include_greeks=False,
        )
        improved = price_option_mc(
            spec=spec,
            config=SimulationConfig(
                num_paths=8_000,
                time_steps=1,
                seed=11,
                antithetic=True,
                control_variate=True,
            ),
            include_greeks=False,
        )

        self.assertLess(improved.standard_error, naive.standard_error)

    def test_up_and_out_call_is_cheaper_than_vanilla_call(self) -> None:
        config = SimulationConfig(
            num_paths=25_000,
            time_steps=126,
            seed=19,
            antithetic=True,
            control_variate=False,
        )
        vanilla = price_option_mc(
            spec=OptionSpec(
                spot=100.0,
                strike=100.0,
                rate=0.03,
                volatility=0.25,
                maturity=1.0,
                payoff=EUROPEAN,
            ),
            config=config,
            include_greeks=False,
        )
        barrier = price_option_mc(
            spec=OptionSpec(
                spot=100.0,
                strike=100.0,
                rate=0.03,
                volatility=0.25,
                maturity=1.0,
                payoff=UP_AND_OUT,
                barrier=125.0,
            ),
            config=config,
            include_greeks=False,
        )

        self.assertLessEqual(barrier.price, vanilla.price)

    def test_zero_volatility_asian_option_is_deterministic(self) -> None:
        spec = OptionSpec(
            spot=105.0,
            strike=100.0,
            rate=0.01,
            volatility=0.0,
            maturity=1.0,
            payoff=ASIAN_ARITHMETIC,
        )
        result = price_option_mc(
            spec=spec,
            config=SimulationConfig(num_paths=2_000, time_steps=12, seed=3),
            include_greeks=False,
        )

        self.assertEqual(result.standard_error, 0.0)
        self.assertEqual(result.confidence_interval[0], result.confidence_interval[1])

    def test_convergence_study_returns_expected_shape(self) -> None:
        spec = OptionSpec(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.2,
            maturity=1.0,
            payoff=EUROPEAN,
        )
        results = run_convergence_study(
            spec=spec,
            path_counts=[2_000, 5_000],
            time_steps=1,
            seed=5,
        )

        self.assertEqual(len(results), 10)
        self.assertEqual(results[0].efficiency_gain, 1.0)
        self.assertIsNotNone(results[0].benchmark_price)

    def test_halton_sampling_prices_vanilla_option_reasonably(self) -> None:
        spec = OptionSpec(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.2,
            maturity=1.0,
            option_type=CALL,
            payoff=EUROPEAN,
        )
        result = price_option_mc(
            spec=spec,
            config=SimulationConfig(
                num_paths=8_000,
                time_steps=1,
                seed=13,
                antithetic=False,
                control_variate=True,
                sampling="halton",
            ),
            include_greeks=False,
        )

        self.assertAlmostEqual(
            result.price,
            black_scholes_price(100.0, 100.0, 0.05, 0.2, 1.0, CALL),
            delta=0.15,
        )

    def test_american_put_has_positive_early_exercise_premium(self) -> None:
        spec = OptionSpec(
            spot=100.0,
            strike=105.0,
            rate=0.05,
            volatility=0.2,
            maturity=1.0,
            option_type=PUT,
            payoff=EUROPEAN,
        )
        result = price_american_option_lsm(
            spec=spec,
            config=SimulationConfig(
                num_paths=25_000,
                time_steps=50,
                seed=17,
                antithetic=True,
                control_variate=False,
            ),
        )

        self.assertGreaterEqual(result.price, result.european_reference_price)
        self.assertGreater(result.early_exercise_ratio, 0.0)

    def test_heston_with_constant_variance_matches_black_scholes(self) -> None:
        spec = HestonSpec(
            spot=100.0,
            strike=100.0,
            rate=0.03,
            maturity=1.0,
            initial_variance=0.04,
            long_run_variance=0.04,
            mean_reversion=1.5,
            vol_of_vol=0.0,
            correlation=-0.7,
            option_type=CALL,
        )
        result = price_heston_option_mc(
            spec=spec,
            config=SimulationConfig(
                num_paths=20_000,
                time_steps=64,
                seed=41,
                antithetic=True,
                control_variate=False,
                sampling="pseudo",
            ),
        )

        self.assertAlmostEqual(
            result.price,
            black_scholes_price(100.0, 100.0, 0.03, 0.2, 1.0, CALL),
            delta=0.18,
        )

    def test_heston_characteristic_function_matches_black_scholes_when_vol_of_vol_is_zero(self) -> None:
        spec = HestonSpec(
            spot=100.0,
            strike=105.0,
            rate=0.03,
            maturity=1.0,
            initial_variance=0.04,
            long_run_variance=0.04,
            mean_reversion=2.0,
            vol_of_vol=0.0,
            correlation=-0.4,
            option_type=CALL,
            dividend_yield=0.0,
        )

        price = price_heston_option_cf(spec, integration_points=600)

        self.assertAlmostEqual(
            price,
            black_scholes_price(100.0, 105.0, 0.03, 0.2, 1.0, CALL),
            delta=0.18,
        )

    def test_heston_smile_exhibits_negative_skew(self) -> None:
        spec = HestonSpec(
            spot=100.0,
            strike=100.0,
            rate=0.03,
            maturity=1.0,
            initial_variance=0.04,
            long_run_variance=0.04,
            mean_reversion=2.0,
            vol_of_vol=0.6,
            correlation=-0.75,
            option_type=CALL,
            dividend_yield=0.01,
        )
        smile = run_heston_smile_cf(
            spec=spec,
            strikes=[80.0, 90.0, 100.0, 110.0, 120.0],
        )

        self.assertEqual(len(smile), 5)
        self.assertIsNotNone(smile[0].implied_volatility)
        self.assertIsNotNone(smile[-1].implied_volatility)
        self.assertGreater(smile[0].implied_volatility, smile[-1].implied_volatility)

    def test_bundled_market_quotes_dataset_loads(self) -> None:
        quotes = load_market_quotes_csv(bundled_market_data_path())

        self.assertEqual(len(quotes), 25)
        self.assertEqual(quotes[0].expiration, "2026-03-16")
        self.assertEqual(quotes[-1].expiration, "2026-05-15")
        self.assertTrue(all(quote.market_implied_volatility is not None for quote in quotes))

    def test_heston_calibration_fits_synthetic_quotes(self) -> None:
        parameters = HestonParameters(
            initial_variance=0.05,
            long_run_variance=0.04,
            mean_reversion=1.8,
            vol_of_vol=0.55,
            correlation=-0.65,
        )
        quotes: list[MarketOptionQuote] = []
        for maturity, expiration in ((0.25, "2026-06-30"), (0.50, "2026-09-30"), (1.0, "2027-03-30")):
            for strike in (90.0, 100.0, 110.0):
                price = price_heston_option_cf(
                    HestonSpec(
                        spot=100.0,
                        strike=strike,
                        rate=0.03,
                        maturity=maturity,
                        initial_variance=parameters.initial_variance,
                        long_run_variance=parameters.long_run_variance,
                        mean_reversion=parameters.mean_reversion,
                        vol_of_vol=parameters.vol_of_vol,
                        correlation=parameters.correlation,
                        option_type=CALL,
                        dividend_yield=0.01,
                    ),
                    integration_points=220,
                )
                quotes.append(
                    MarketOptionQuote(
                        snapshot_date="2026-03-30",
                        expiration=expiration,
                        underlying="TEST",
                        spot=100.0,
                        strike=strike,
                        maturity=maturity,
                        rate=0.03,
                        dividend_yield=0.01,
                        option_type=CALL,
                        bid=price,
                        ask=price,
                        market_price=price,
                        market_implied_volatility=implied_volatility_from_price(
                            price,
                            spot=100.0,
                            strike=strike,
                            rate=0.03,
                            maturity=maturity,
                            option_type=CALL,
                            dividend_yield=0.01,
                        ),
                    )
                )

        result = calibrate_heston_parameters(
            quotes,
            seed=5,
            global_samples=18,
            search_rounds=4,
            local_samples=12,
            integration_points=144,
        )

        self.assertLess(result.rmse_price, 0.35)
        self.assertLess(result.max_abs_price_error, 0.65)
        self.assertIsNotNone(result.rmse_implied_volatility)
        self.assertLess(result.rmse_implied_volatility, 0.025)
        self.assertLess(result.parameters.correlation, -0.15)
        self.assertGreater(result.parameters.vol_of_vol, 0.15)

    def test_geometric_basket_mc_tracks_closed_form(self) -> None:
        spec = BasketOptionSpec(
            spots=(100.0, 95.0, 105.0),
            volatilities=(0.20, 0.25, 0.22),
            weights=(0.4, 0.3, 0.3),
            correlation_matrix=build_equicorrelation_matrix(3, 0.35),
            strike=100.0,
            rate=0.04,
            maturity=1.0,
            option_type=CALL,
            payoff=BASKET_GEOMETRIC,
        )
        result = price_basket_option_mc(
            spec=spec,
            config=SimulationConfig(
                num_paths=40_000,
                time_steps=1,
                seed=23,
                antithetic=True,
                control_variate=False,
            ),
            use_geometric_control_variate=False,
        )

        self.assertAlmostEqual(result.price, geometric_basket_price(spec), delta=0.12)

    def test_basket_control_variate_reduces_standard_error(self) -> None:
        spec = BasketOptionSpec(
            spots=(100.0, 95.0, 105.0),
            volatilities=(0.20, 0.25, 0.22),
            weights=(0.4, 0.3, 0.3),
            correlation_matrix=build_equicorrelation_matrix(3, 0.35),
            strike=100.0,
            rate=0.04,
            maturity=1.0,
            option_type=CALL,
            payoff=BASKET_ARITHMETIC,
        )
        base_config = SimulationConfig(
            num_paths=12_000,
            time_steps=1,
            seed=29,
            antithetic=True,
            control_variate=False,
        )
        without_cv = price_basket_option_mc(
            spec=spec,
            config=base_config,
            use_geometric_control_variate=False,
        )
        with_cv = price_basket_option_mc(
            spec=spec,
            config=base_config,
            use_geometric_control_variate=True,
        )

        self.assertLess(with_cv.standard_error, without_cv.standard_error)

    def test_surface_returns_full_grid(self) -> None:
        spec = OptionSpec(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.2,
            maturity=1.0,
            payoff=EUROPEAN,
        )
        points = run_sensitivity_surface(
            spec=spec,
            config=SimulationConfig(
                num_paths=8_000,
                time_steps=1,
                seed=31,
                antithetic=True,
                control_variate=True,
            ),
            spots=[90.0, 100.0, 110.0],
            volatilities=[0.15, 0.25],
            include_greeks=True,
        )

        self.assertEqual(len(points), 6)
        self.assertIsNotNone(points[0].delta)

    def test_research_report_writes_markdown_summary(self) -> None:
        output_path = Path("outputs") / "test_report.md"
        try:
            build_research_report(
                str(output_path),
                seed=3,
                study_paths=[500, 1_000],
                surface_spots=[95.0, 100.0],
                surface_vols=[0.2],
                american_paths=2_000,
                basket_paths=2_000,
                surface_paths=1_000,
                heston_paths=2_000,
                calibration_data=str(bundled_market_data_path()),
            )
            contents = output_path.read_text(encoding="utf-8")
        finally:
            if output_path.exists():
                output_path.unlink()

        self.assertIn("European Option Convergence Study", contents)
        self.assertIn("American Put via Longstaff-Schwartz", contents)
        self.assertIn("Correlated Basket Option", contents)
        self.assertIn("Heston Volatility Smile", contents)
        self.assertIn("Heston Calibration to a Market Snapshot", contents)


if __name__ == "__main__":
    unittest.main()
