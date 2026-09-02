"""Regression tests for the planar-purity analysis."""

from __future__ import annotations

import math
import unittest

import numpy as np

import purity_analysis as pa


class PurityAnalysisTests(unittest.TestCase):
    def test_planar_subsystem_convention(self) -> None:
        self.assertEqual(len(pa.planar_subsystems(8, unique=True)), 4)
        self.assertEqual(len(pa.planar_subsystems(8, unique=False)), 8)
        self.assertEqual(len(pa.planar_subsystems(7, unique=True)), 7)
        self.assertEqual(len(set(pa.planar_subsystems(7))), 7)
        self.assertEqual(len(pa.planar_subsystems(8, block_size=3)), 8)
        self.assertEqual(
            pa.planar_subsystems(8, block_size=3)[0], (0, 1, 2)
        )

    def test_exact_mean(self) -> None:
        self.assertAlmostEqual(pa.exact_planar_mean(4), 8 / 17)
        self.assertAlmostEqual(pa.exact_planar_mean(5), 12 / 33)
        self.assertAlmostEqual(
            pa.exact_planar_mean_local_dimension(4, 3), 18 / 82
        )
        self.assertAlmostEqual(
            pa.exact_planar_mean_local_dimension(6, 3, block_size=2),
            (3**2 + 3**4) / (3**6 + 1),
        )

    def test_variance_matches_even_closed_form(self) -> None:
        for n in (2, 4, 6, 8, 10):
            dimension = 2**n
            closed_form = (
                20 * (dimension**2 - 1) / (3 * n) - 8 * dimension
            ) / ((dimension + 1) ** 2 * (dimension + 2) * (dimension + 3))
            self.assertAlmostEqual(pa.exact_planar_variance(n), closed_form, places=15)

    def test_variance_matches_odd_closed_form(self) -> None:
        for n in (3, 5, 7, 9):
            dimension = 2**n
            closed_form = (
                6 * (dimension**2 - 1) / n - 9 * dimension
            ) / ((dimension + 1) ** 2 * (dimension + 2) * (dimension + 3))
            self.assertAlmostEqual(pa.exact_planar_variance(n), closed_form, places=15)

    def test_qudit_variance_matches_overlap_sum(self) -> None:
        for local_dimension in (2, 3, 4):
            for n in range(2, 9):
                mean = pa.exact_planar_mean_local_dimension(n, local_dimension)
                direct_variance = (
                    pa.exact_planar_second_moment_local_dimension(
                        n, local_dimension
                    )
                    - mean**2
                )
                self.assertAlmostEqual(
                    direct_variance,
                    pa.exact_planar_variance_local_dimension(
                        n, local_dimension
                    ),
                    places=13,
                )

    def test_general_k_variance_matches_overlap_sum(self) -> None:
        for local_dimension in (2, 3):
            for n in range(4, 9):
                for block_size in range(1, n // 2 + 1):
                    mean = pa.exact_planar_mean_local_dimension(
                        n, local_dimension, block_size
                    )
                    direct_variance = (
                        pa.exact_planar_second_moment_local_dimension(
                            n, local_dimension, block_size
                        )
                        - mean**2
                    )
                    self.assertAlmostEqual(
                        direct_variance,
                        pa.exact_planar_variance_local_dimension(
                            n, local_dimension, block_size
                        ),
                        places=13,
                    )
                    self.assertAlmostEqual(
                        pa.exact_planar_second_moment_local_dimension(
                            n, local_dimension, block_size
                        ),
                        pa.exact_planar_second_moment_closed_local_dimension(
                            n, local_dimension, block_size
                        ),
                        places=15,
                    )

    def test_trace_polynomial_matches_direct_s4_enumeration(self) -> None:
        """Audit all 24 four-replica contributions, not only their sum formula."""

        permutations = list(__import__("itertools").permutations(range(4)))
        identity = tuple(range(4))
        swap_01 = (1, 0, 2, 3)
        swap_23 = (0, 1, 3, 2)
        both_swaps = (1, 0, 3, 2)
        for n in range(2, 11):
            for block_size in range(1, n // 2 + 1):
                for overlap in range(block_size + 1):
                    dimensions = (
                        2**overlap,
                        2 ** (block_size - overlap),
                        2 ** (block_size - overlap),
                        2 ** (n - 2 * block_size + overlap),
                    )
                    local_actions = (both_swaps, swap_01, swap_23, identity)
                    direct_sum = sum(
                        math.prod(
                            dimension
                            ** pa._cycle_count(pa._compose(local, permutation))
                            for dimension, local in zip(dimensions, local_actions)
                        )
                        for permutation in permutations
                    )
                    self.assertEqual(
                        direct_sum,
                        pa._second_moment_trace_polynomial(
                            n, overlap, block_size
                        ),
                    )

    def test_delta_diagram_component_counts(self) -> None:
        """Check the two representative contractions drawn in Figs. 2 and 3."""

        for n in range(2, 11):
            for block_size in range(1, n // 2 + 1):
                for overlap in range(block_size + 1):
                    x = 2**overlap
                    y = 2 ** (block_size - overlap)
                    w = 2 ** (n - 2 * block_size + overlap)
                    region_dimensions = (y, x, y, w)

                    three_replica_components = (2, 1, 2, 3)
                    three_replica_count = math.prod(
                        dimension**components
                        for dimension, components in zip(
                            region_dimensions, three_replica_components
                        )
                    )
                    self.assertEqual(three_replica_count, x * y**4 * w**3)

                    crossed_components = (1, 2, 1, 2)
                    crossed_count = math.prod(
                        dimension**components
                        for dimension, components in zip(
                            region_dimensions, crossed_components
                        )
                    )
                    self.assertEqual(crossed_count, x**2 * y**2 * w**2)

    def test_pair_covariances_average_to_planar_variance(self) -> None:
        for n in range(2, 13):
            for block_size in range(1, n // 2 + 1):
                covariance_average = sum(
                    pa.exact_planar_pair_covariance(
                        n, displacement, block_size
                    )
                    for displacement in range(n)
                ) / n
                self.assertAlmostEqual(
                    covariance_average,
                    pa.exact_planar_variance(n, block_size),
                    places=15,
                )

    def test_pair_kernel_recovers_absolute_variance(self) -> None:
        for n in (2, 4, 6, 8, 10, 12):
            from_kernel = float(
                pa.exact_absolute_variance_even_from_pair_kernel(n)
            )
            self.assertAlmostEqual(
                from_kernel,
                pa.exact_absolute_variance_even(n),
                places=15,
            )

    def test_third_moment_regression(self) -> None:
        self.assertAlmostEqual(pa.exact_planar_skewness(4), 0.667698, places=5)
        for n in (4, 5, 6):
            exact_fraction = pa.exact_planar_third_cumulant_fraction(n)
            self.assertAlmostEqual(
                float(exact_fraction),
                pa.exact_planar_third_cumulant(n),
                places=18,
            )

    def test_batch_purities_are_physical(self) -> None:
        rng = np.random.default_rng(20260719)
        states = pa.haar_state_batch(4, 32, rng)
        planar_unique = pa.averaged_purity_batch(
            states, 4, pa.planar_subsystems(4, unique=True)
        )
        planar_duplicated = pa.averaged_purity_batch(
            states, 4, pa.planar_subsystems(4, unique=False)
        )
        self.assertTrue(np.all(planar_unique >= 1 / 4 - 1e-14))
        self.assertTrue(np.all(planar_unique <= 1 + 1e-14))
        np.testing.assert_allclose(planar_unique, planar_duplicated, atol=2e-15)

        qutrit_states = pa.haar_state_batch(
            4, 16, rng, local_dimension=3
        )
        qutrit_planar = pa.averaged_purity_batch(
            qutrit_states,
            4,
            pa.planar_subsystems(4, block_size=2),
            local_dimension=3,
        )
        self.assertTrue(np.all(qutrit_planar >= 1 / 9 - 1e-14))
        self.assertTrue(np.all(qutrit_planar <= 1 + 1e-14))

    def test_fixed_k_asymptotic_coefficient(self) -> None:
        for local_dimension, block_size in ((2, 1), (2, 3), (3, 2)):
            p = local_dimension
            k = block_size
            coefficient = (
                p ** (2 * k)
                + 2 * (p ** (2 * k) - p**2) / (p**2 - 1)
                - 2 * k
                + 1
            )
            n = 40
            dimension = p**n
            leading = 2 * coefficient / (n * p ** (2 * k) * dimension**2)
            exact = pa.exact_planar_variance_local_dimension(n, p, k)
            self.assertLess(abs(exact / leading - 1), 1e-6)

    def test_extensive_k_asymptotic_coefficient(self) -> None:
        for local_dimension in (2, 3):
            p = local_dimension
            n = 100
            block_size = n // 4
            dimension = p**n
            leading = (
                2
                * (p**2 + 1)
                / ((p**2 - 1) * n * dimension**2)
            )
            exact = pa.exact_planar_variance_local_dimension(
                n, p, block_size
            )
            self.assertLess(abs(exact / leading - 1), 1e-6)

    def test_saved_samples_have_expected_size(self) -> None:
        for n in (4, 5, 6, 7, 8, 10):
            self.assertEqual(pa.load_saved_planar(n).size, 40_000)
        self.assertEqual(pa.load_saved_absolute(10).size, 40_000)
        self.assertEqual(pa.load_seeded_balanced(9).size, 40_000)
        self.assertEqual(pa.load_seeded_balanced(10).size, 120_000)
        for n, local_dimension, block_sizes in (
            (6, 2, (1, 2, 3)),
            (4, 3, (1, 2)),
        ):
            for block_size in block_sizes:
                self.assertEqual(
                    pa.load_general_validation(
                        n, local_dimension, block_size
                    ).size,
                    40_000,
                )

    def test_uncertainty_estimates_are_finite(self) -> None:
        statistics = pa.sample_statistics(pa.load_saved_planar(4))
        for key in ("mean_se", "variance_se", "skewness_se"):
            self.assertTrue(math.isfinite(statistics[key]))
            self.assertGreater(statistics[key], 0)


if __name__ == "__main__":
    unittest.main()
