#!/usr/bin/env python3
"""Reproducible analysis for planar k-purity and absolute purity.

The script provides:

* efficient Haar-state sampling (normalized complex Gaussian vectors);
* planar k-purity and absolute balanced-purity calculations;
* exact planar mean and variance for every 1 <= k <= floor(n/2);
* exact balanced-case third cumulant and standardized skewness;
* arbitrary local dimension in the exact first- and second-moment formulas;
* uncertainty-aware comparison with the saved Monte Carlo samples; and
* publication-quality PDF/PNG figures used by the manuscript.

Run from the project root, for example::

    python code/purity_analysis.py summarize
    python code/purity_analysis.py simulate-comparisons --samples 40000
    python code/purity_analysis.py figures
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kurtosis, norm, skew


# Embed scalable TrueType glyphs in PDF figures instead of Type 3 bitmap fonts.
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
FIGURE_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"


def _validated_block_size(n: int, block_size: int | None) -> int:
    """Return a valid planar block size, using floor(n/2) by default."""

    if n < 2:
        raise ValueError("n must be at least 2.")
    if block_size is None:
        return n // 2
    if not 1 <= block_size <= n // 2:
        raise ValueError("block_size must satisfy 1 <= block_size <= floor(n/2).")
    return block_size


def planar_subsystems(
    n: int, unique: bool = True, block_size: int | None = None
) -> list[tuple[int, ...]]:
    """Return contiguous k-site subsystems on an n-cycle.

    Only in the balanced even case, k=n/2, do complementary blocks have
    identical purity for a global pure state.  There ``unique=True`` retains
    one block from each complementary pair.  Otherwise all n cyclic blocks
    define distinct bipartitions.
    """

    k = _validated_block_size(n, block_size)
    balanced_even = n % 2 == 0 and k == n // 2
    count = n // 2 if unique and balanced_even else n
    return [
        tuple(sorted((start + j) % n for j in range(k)))
        for start in range(count)
    ]


def absolute_subsystems(n: int) -> list[tuple[int, ...]]:
    """Return one representative of every balanced bipartition."""

    r = n // 2
    all_subsystems = list(itertools.combinations(range(n), r))
    if n % 2:
        return all_subsystems

    representatives: list[tuple[int, ...]] = []
    universe = set(range(n))
    for subsystem in all_subsystems:
        complement = tuple(sorted(universe.difference(subsystem)))
        if subsystem <= complement:
            representatives.append(subsystem)
    return representatives


def haar_state_batch(
    n: int,
    batch_size: int,
    rng: np.random.Generator,
    local_dimension: int = 2,
) -> np.ndarray:
    """Generate Haar-random n-party pure states of local dimension p."""

    if local_dimension < 2:
        raise ValueError("local_dimension must be at least 2.")
    dimension = local_dimension**n
    states = rng.normal(size=(batch_size, dimension)) + 1j * rng.normal(
        size=(batch_size, dimension)
    )
    states /= np.linalg.norm(states, axis=1, keepdims=True)
    return states


def subsystem_purity_batch(
    states: np.ndarray,
    n: int,
    keep: Sequence[int],
    local_dimension: int = 2,
) -> np.ndarray:
    """Compute subsystem purity for each state without forming the full density matrix."""

    if local_dimension < 2:
        raise ValueError("local_dimension must be at least 2.")
    keep = tuple(sorted(keep))
    traced = tuple(i for i in range(n) if i not in keep)
    expected_dimension = local_dimension**n
    if states.ndim != 2 or states.shape[1] != expected_dimension:
        raise ValueError(
            "states must have shape (batch_size, local_dimension**n)."
        )
    tensor = states.reshape((states.shape[0],) + (local_dimension,) * n)
    permutation = (0,) + tuple(i + 1 for i in keep + traced)
    matrix = tensor.transpose(permutation).reshape(
        states.shape[0],
        local_dimension ** len(keep),
        local_dimension ** len(traced),
    )
    reduced = matrix @ matrix.conj().transpose(0, 2, 1)
    return np.einsum("bij,bji->b", reduced, reduced, optimize=True).real


def averaged_purity_batch(
    states: np.ndarray,
    n: int,
    subsystems: Iterable[Sequence[int]],
    local_dimension: int = 2,
) -> np.ndarray:
    """Average subsystem purities over a supplied collection of bipartitions."""

    subsystems = list(subsystems)
    values = np.zeros(states.shape[0], dtype=float)
    for subsystem in subsystems:
        values += subsystem_purity_batch(
            states, n, subsystem, local_dimension=local_dimension
        )
    return values / len(subsystems)


def simulate_planar_distributions(
    n: int,
    local_dimension: int,
    block_sizes: Sequence[int],
    samples: int,
    seed: int,
    batch_size: int = 256,
) -> dict[int, np.ndarray]:
    """Sample several planar k-purities on the same Haar-random states."""

    if samples < 1:
        raise ValueError("samples must be positive.")
    validated_sizes = tuple(
        dict.fromkeys(_validated_block_size(n, size) for size in block_sizes)
    )
    if not validated_sizes:
        raise ValueError("at least one block size is required.")

    rng = np.random.default_rng(seed)
    outputs = {
        size: np.empty(samples, dtype=float) for size in validated_sizes
    }
    subsystems = {
        size: planar_subsystems(n, unique=True, block_size=size)
        for size in validated_sizes
    }
    for start in range(0, samples, batch_size):
        stop = min(start + batch_size, samples)
        states = haar_state_batch(
            n,
            stop - start,
            rng,
            local_dimension=local_dimension,
        )
        for size in validated_sizes:
            outputs[size][start:stop] = averaged_purity_batch(
                states,
                n,
                subsystems[size],
                local_dimension=local_dimension,
            )
    return outputs


def simulate_joint_distributions(
    n: int,
    samples: int,
    seed: int,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample planar and absolute purity on the same Haar-random states."""

    rng = np.random.default_rng(seed)
    planar = np.empty(samples, dtype=float)
    absolute = np.empty(samples, dtype=float)
    p_subsystems = planar_subsystems(n, unique=True)
    a_subsystems = absolute_subsystems(n)

    for start in range(0, samples, batch_size):
        stop = min(start + batch_size, samples)
        states = haar_state_batch(n, stop - start, rng)
        planar[start:stop] = averaged_purity_batch(states, n, p_subsystems)
        absolute[start:stop] = averaged_purity_batch(states, n, a_subsystems)
    return planar, absolute


def exact_planar_mean_local_dimension(
    n: int, local_dimension: int, block_size: int | None = None
) -> float:
    """Exact Haar mean of planar k-purity for local dimension p."""

    k = _validated_block_size(n, block_size)
    if local_dimension < 2:
        raise ValueError("local_dimension must be at least 2.")
    dimension = local_dimension**n
    return (
        local_dimension**k + local_dimension ** (n - k)
    ) / (dimension + 1)


def _exact_planar_mean_fraction(
    n: int, local_dimension: int = 2, block_size: int | None = None
) -> Fraction:
    """Exact Haar mean as a rational number."""

    k = _validated_block_size(n, block_size)
    if local_dimension < 2:
        raise ValueError("local_dimension must be at least 2.")
    dimension = local_dimension**n
    return Fraction(
        local_dimension**k + local_dimension ** (n - k),
        dimension + 1,
    )


def exact_planar_mean(n: int, block_size: int | None = None) -> float:
    """Exact Haar mean of planar k-purity for qubits."""

    return exact_planar_mean_local_dimension(n, 2, block_size)


def cyclic_overlap(
    n: int, displacement: int, block_size: int | None = None
) -> int:
    """Overlap of two cyclic k-site blocks at a relative displacement."""

    k = _validated_block_size(n, block_size)
    displacement %= n
    return max(0, k - displacement) + max(0, k - (n - displacement))


def _second_moment_trace_polynomial_local_dimension(
    n: int,
    overlap: int,
    local_dimension: int,
    block_size: int | None = None,
) -> int:
    """Permutation trace sum for two subsystem purities.

    The four disjoint regions have dimensions x, y, y, and w, corresponding to
    A intersect C, A minus C, C minus A, and the complement of A union C.
    """

    k = _validated_block_size(n, block_size)
    if not 0 <= overlap <= k:
        raise ValueError("overlap must satisfy 0 <= overlap <= block_size.")
    x = local_dimension**overlap
    y = local_dimension ** (k - overlap)
    w = local_dimension ** (n - 2 * k + overlap)
    return (
        2 * x * y**4 * w
        + 4 * x * y**4 * w**3
        + 2 * x**2 * y**2 * w**2
        + 8 * x**2 * y**4 * w**2
        + x**2 * y**6 * w**4
        + 4 * x**3 * y**4 * w
        + 2 * x**3 * y**6 * w**3
        + x**4 * y**6 * w**2
    )


def _second_moment_trace_polynomial(
    n: int, overlap: int, block_size: int | None = None
) -> int:
    """Permutation trace sum for two qubit subsystem purities."""

    return _second_moment_trace_polynomial_local_dimension(
        n, overlap, 2, block_size
    )


def exact_planar_second_moment_local_dimension(
    n: int, local_dimension: int, block_size: int | None = None
) -> float:
    """Exact Haar second moment of planar k-purity."""

    k = _validated_block_size(n, block_size)
    if local_dimension < 2:
        raise ValueError("local_dimension must be at least 2.")
    dimension = local_dimension**n
    denominator = n * math.prod(dimension + j for j in range(4))
    numerator = sum(
        _second_moment_trace_polynomial_local_dimension(
            n,
            cyclic_overlap(n, displacement, k),
            local_dimension,
            k,
        )
        for displacement in range(n)
    )
    return numerator / denominator


def exact_planar_second_moment(
    n: int, block_size: int | None = None
) -> float:
    """Exact Haar second moment of planar k-purity for qubits."""

    return exact_planar_second_moment_local_dimension(n, 2, block_size)


def _exact_planar_second_moment_closed_fraction(
    n: int, local_dimension: int, block_size: int | None = None
) -> Fraction:
    """Exact general-k closed form corresponding to Eqs. (25)--(29)."""

    k = _validated_block_size(n, block_size)
    if local_dimension < 2:
        raise ValueError("local_dimension must be at least 2.")
    p = local_dimension
    dimension = p**n
    a = p**k
    b = p ** (n - k)
    trace_constant = (
        2 * dimension**3
        + 8 * dimension**2
        + (dimension**2 + 4 * dimension) * (a**2 + b**2)
    )
    zero_overlap_count = n - 2 * k + 1
    sum_positive = (
        Fraction(a**2)
        + Fraction(2 * (a**2 - p**2), p**2 - 1)
        + zero_overlap_count
    )
    sum_negative = (
        Fraction(1, a**2)
        + 2
        * (Fraction(1) - Fraction(1, p ** (2 * (k - 1))))
        / (p**2 - 1)
        + zero_overlap_count
    )
    averaged_trace = (
        Fraction(trace_constant)
        + Fraction(2 * dimension * a**2, n) * sum_negative
        + Fraction(2 * b**2, n) * sum_positive
    )
    return averaged_trace / math.prod(dimension + j for j in range(4))


def exact_planar_second_moment_closed_local_dimension(
    n: int, local_dimension: int, block_size: int | None = None
) -> float:
    """Exact closed second moment after evaluating both geometric sums."""

    return float(
        _exact_planar_second_moment_closed_fraction(
            n, local_dimension, block_size
        )
    )


def exact_planar_variance_local_dimension(
    n: int, local_dimension: int, block_size: int | None = None
) -> float:
    """Exact Haar variance of planar k-purity.

    The compact parity formulas are used for balanced blocks.  The general-k
    branch evaluates the exact overlap sum before converting to floating point.
    """

    k = _validated_block_size(n, block_size)
    if local_dimension < 2:
        raise ValueError("local_dimension must be at least 2.")
    p = local_dimension
    dimension = p**n
    if k != n // 2:
        mean = Fraction(p**k + p ** (n - k), dimension + 1)
        variance = _exact_planar_second_moment_closed_fraction(
            n, p, k
        ) - mean**2
        return float(variance)

    denominator = (dimension + 1) ** 2 * (dimension + 2) * (dimension + 3)
    if n % 2 == 0:
        numerator = (
            4
            * (p**2 + 1)
            * (dimension**2 - 1)
            / (n * (p**2 - 1))
            - 8 * dimension
        )
    else:
        numerator = (
            2
            * (p + 1)
            * (dimension**2 - 1)
            / (n * (p - 1))
            - 2 * (p + 1) ** 2 * dimension / p
        )
    return numerator / denominator


def exact_planar_variance(n: int, block_size: int | None = None) -> float:
    """Exact Haar variance of planar k-purity for qubits."""

    return exact_planar_variance_local_dimension(n, 2, block_size)


def exact_planar_pair_covariance(
    n: int,
    displacement: int,
    block_size: int | None = None,
    local_dimension: int = 2,
) -> float:
    """Covariance of two cyclic k-block purities at relative displacement."""

    k = _validated_block_size(n, block_size)
    if local_dimension < 2:
        raise ValueError("local_dimension must be at least 2.")
    dimension = local_dimension**n
    overlap = cyclic_overlap(n, displacement, k)
    joint_moment = _second_moment_trace_polynomial_local_dimension(
        n, overlap, local_dimension, k
    ) / math.prod(
        dimension + j for j in range(4)
    )
    return joint_moment - exact_planar_mean_local_dimension(
        n, local_dimension, k
    ) ** 2


def exact_absolute_variance_even(n: int) -> float:
    """Known exact variance of balanced absolute purity for even n."""

    if n % 2:
        raise ValueError("The closed form implemented here assumes even n.")
    r = n // 2
    dimension = 2**n
    subsystem_dimension = 2**r
    f2 = 2 / math.comb(n, r) * sum(
        math.comb(r, d)
        * math.comb(r, d)
        * 2 ** (n / 2)
        * (4 ** (n / 4 - d) + 4 ** (-(n / 4 - d)))
        for d in range(r + 1)
    )
    return (
        (dimension + 1) * f2 - 2 * (2 * subsystem_dimension) ** 2
    ) / ((dimension + 1) ** 2 * (dimension + 2) * (dimension + 3))


def exact_absolute_second_moment_even_from_pair_kernel(n: int) -> Fraction:
    """Recover the even-n absolute-purity second moment from the pair kernel.

    The absolute balanced functional may be averaged over all r-subsets because
    complementary subsets have identical purity.  For a fixed subset A, there
    are binom(r, d)^2 subsets B with |A minus B|=d and hence overlap r-d.
    Keeping the result rational makes this an exact cross-check of the cyclic
    calculation against the established all-balanced-cut formula.
    """

    if n % 2:
        raise ValueError("The all-balanced-cut identity implemented here assumes even n.")
    r = n // 2
    dimension = 2**n
    numerator = sum(
        math.comb(r, displacement) ** 2
        * _second_moment_trace_polynomial(
            n, r - displacement, block_size=r
        )
        for displacement in range(r + 1)
    )
    denominator = math.comb(n, r) * math.prod(
        dimension + j for j in range(4)
    )
    return Fraction(numerator, denominator)


def exact_absolute_variance_even_from_pair_kernel(n: int) -> Fraction:
    """Recover the even-n absolute-purity variance from the pair kernel."""

    mean = _exact_planar_mean_fraction(n, 2, n // 2)
    return exact_absolute_second_moment_even_from_pair_kernel(n) - mean**2


def _compose(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(left[right[i]] for i in range(len(left)))


def _cycle_count(permutation: tuple[int, ...]) -> int:
    visited: set[int] = set()
    cycles = 0
    for start in range(len(permutation)):
        if start in visited:
            continue
        cycles += 1
        current = start
        while current not in visited:
            visited.add(current)
            current = permutation[current]
    return cycles


def exact_planar_third_moment_fraction(
    n: int, block_size: int | None = None
) -> Fraction:
    """Exact Haar third raw moment as a rational six-replica sum."""

    k = _validated_block_size(n, block_size)
    dimension = 2**n
    blocks = [set((start + j) % n for j in range(k)) for start in range(n)]
    permutations = list(itertools.permutations(range(6)))
    swaps = ((0, 1), (2, 3), (4, 5))

    local_permutations: list[tuple[int, ...]] = []
    for mask in range(8):
        local = list(range(6))
        for bit, (left, right) in enumerate(swaps):
            if (mask >> bit) & 1:
                local[left], local[right] = local[right], local[left]
        local_permutations.append(tuple(local))

    cycle_table = [
        [
            _cycle_count(_compose(local_permutations[mask], permutation))
            for mask in range(8)
        ]
        for permutation in permutations
    ]

    numerator = 0
    for first in blocks:
        for second in blocks:
            for third in blocks:
                membership_counts = [0] * 8
                for site in range(n):
                    mask = (
                        int(site in first)
                        + 2 * int(site in second)
                        + 4 * int(site in third)
                    )
                    membership_counts[mask] += 1
                numerator += sum(
                    2
                    ** sum(
                        cycles[mask] * membership_counts[mask]
                        for mask in range(8)
                    )
                    for cycles in cycle_table
                )

    denominator = n**3 * math.prod(dimension + j for j in range(6))
    return Fraction(numerator, denominator)


def exact_planar_third_moment(
    n: int, block_size: int | None = None
) -> float:
    """Floating-point value of the exact Haar third raw moment."""

    return float(exact_planar_third_moment_fraction(n, block_size))


def exact_planar_third_cumulant_fraction(
    n: int, block_size: int | None = None
) -> Fraction:
    """Exact centered third moment as a rational number."""

    mean = _exact_planar_mean_fraction(n, 2, block_size)
    second = _exact_planar_second_moment_closed_fraction(
        n, 2, block_size
    )
    third = exact_planar_third_moment_fraction(n, block_size)
    return third - 3 * second * mean + 2 * mean**3


def exact_planar_third_cumulant(
    n: int, block_size: int | None = None
) -> float:
    return float(exact_planar_third_cumulant_fraction(n, block_size))


def exact_planar_skewness(n: int, block_size: int | None = None) -> float:
    variance = exact_planar_variance(n, block_size)
    return exact_planar_third_cumulant(n, block_size) / variance ** 1.5


def load_saved_planar(n: int) -> np.ndarray:
    return np.loadtxt(DATA_DIR / f"purity_samples_p(n={n})")


def load_saved_absolute(n: int) -> np.ndarray:
    return np.loadtxt(DATA_DIR / f"purity_samples_a(n={n})")


def load_seeded_balanced(n: int) -> np.ndarray:
    """Load the fixed-seed balanced sample used for n=9 or the n=10 rerun."""

    if n not in (9, 10):
        raise ValueError("fixed-seed balanced samples are available for n=9 and 10.")
    loaded = np.load(RESULTS_DIR / f"validation_balanced_n{n}.npz")
    return loaded[f"k{n // 2}"]


def load_manuscript_planar(n: int) -> np.ndarray:
    """Load the planar sample used in the main validation table."""

    if n in (9, 10):
        return load_seeded_balanced(n)
    return load_saved_planar(n)


def load_general_validation(
    n: int, local_dimension: int, block_size: int
) -> np.ndarray:
    """Load a fixed-seed nonbalanced/qudit validation sample."""

    path = RESULTS_DIR / f"validation_general_p{local_dimension}_n{n}.npz"
    loaded = np.load(path)
    return loaded[f"k{block_size}"]


def sample_statistics(values: np.ndarray) -> dict[str, float]:
    """Sample moments and asymptotic one-standard-error estimates.

    The variance uses Bessel's correction.  The skewness standard error is
    obtained from the empirical influence function of the population
    standardized third central moment.  The distinction between that
    functional and the finite-sample corrected skewness is O(1/M).
    """

    values = np.asarray(values, dtype=float)
    sample_count = values.size
    centered = values - np.mean(values)
    moment_2 = float(np.mean(centered**2))
    moment_3 = float(np.mean(centered**3))
    moment_4 = float(np.mean(centered**4))
    unbiased_variance = float(np.var(values, ddof=1))

    mean_se = math.sqrt(unbiased_variance / sample_count)
    variance_se = math.sqrt(max(moment_4 - moment_2**2, 0.0) / sample_count)

    influence_2 = centered**2 - moment_2
    influence_3 = centered**3 - moment_3 - 3 * moment_2 * centered
    influence_skewness = (
        influence_3 / moment_2**1.5
        - 1.5 * moment_3 * influence_2 / moment_2**2.5
    )
    skewness_se = math.sqrt(
        float(np.mean(influence_skewness**2)) / sample_count
    )

    return {
        "samples": int(sample_count),
        "mean": float(np.mean(values)),
        "mean_se": mean_se,
        "variance": unbiased_variance,
        "variance_se": variance_se,
        "std_dev": math.sqrt(unbiased_variance),
        "skewness": float(skew(values, bias=False)),
        "skewness_se": skewness_se,
        "excess_kurtosis": float(kurtosis(values, bias=False)),
    }


def write_summaries() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    available_n = [4, 5, 6, 7, 8, 9, 10]
    sample_rows: list[dict[str, float | int | str]] = []
    theory_rows: list[dict[str, float | int]] = []

    for n in available_n:
        values = load_manuscript_planar(n)
        row: dict[str, float | int | str] = {"functional": "planar", "n": n}
        row.update(sample_statistics(values))
        sample_rows.append(row)

        theory_rows.append(
            {
                "n": n,
                "mean": exact_planar_mean(n),
                "variance": exact_planar_variance(n),
                "std_dev": math.sqrt(exact_planar_variance(n)),
                "third_cumulant": exact_planar_third_cumulant(n),
                "skewness": exact_planar_skewness(n),
            }
        )

    absolute_values = load_saved_absolute(10)
    absolute_row: dict[str, float | int | str] = {
        "functional": "absolute",
        "n": 10,
    }
    absolute_row.update(sample_statistics(absolute_values))
    sample_rows.append(absolute_row)

    with (RESULTS_DIR / "sample_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sample_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sample_rows)

    with (RESULTS_DIR / "theory_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(theory_rows[0].keys()))
        writer.writeheader()
        writer.writerows(theory_rows)

    exact_third_rows: list[dict[str, int | float]] = []
    for n in available_n:
        raw = exact_planar_third_moment_fraction(n)
        centered = exact_planar_third_cumulant_fraction(n)
        exact_third_rows.append(
            {
                "n": n,
                "raw_third_numerator": raw.numerator,
                "raw_third_denominator": raw.denominator,
                "centered_third_numerator": centered.numerator,
                "centered_third_denominator": centered.denominator,
                "standardized_skewness": exact_planar_skewness(n),
            }
        )
    with (RESULTS_DIR / "exact_third_moments.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(exact_third_rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(exact_third_rows)

    general_rows: list[dict[str, int | float]] = []
    for n, local_dimension, block_sizes in (
        (6, 2, (1, 2, 3)),
        (4, 3, (1, 2)),
    ):
        for block_size in block_sizes:
            values = load_general_validation(
                n, local_dimension, block_size
            )
            statistics = sample_statistics(values)
            exact_mean = exact_planar_mean_local_dimension(
                n, local_dimension, block_size
            )
            exact_variance = exact_planar_variance_local_dimension(
                n, local_dimension, block_size
            )
            general_rows.append(
                {
                    "n": n,
                    "p": local_dimension,
                    "k": block_size,
                    "samples": int(statistics["samples"]),
                    "exact_mean": exact_mean,
                    "sample_mean": statistics["mean"],
                    "mean_residual_se": (
                        statistics["mean"] - exact_mean
                    )
                    / statistics["mean_se"],
                    "exact_variance": exact_variance,
                    "sample_variance": statistics["variance"],
                    "variance_residual_se": (
                        statistics["variance"] - exact_variance
                    )
                    / statistics["variance_se"],
                }
            )
    with (RESULTS_DIR / "general_validation.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(general_rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(general_rows)

    print("Saved sample and exact-theory summaries in", RESULTS_DIR)
    for sample_row, theory_row in zip(sample_rows[: len(available_n)], theory_rows):
        print(
            f"n={sample_row['n']}: "
            f"sample mean={sample_row['mean']:.10g}, exact={theory_row['mean']:.10g}; "
            f"sample var={sample_row['variance']:.10g}, exact={theory_row['variance']:.10g}; "
            f"sample skew={sample_row['skewness']:.6g}, exact={theory_row['skewness']:.6g}"
        )


def simulate_comparisons(samples: int, batch_size: int) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for n, seed in ((4, 4104), (8, 4108)):
        print(f"Sampling n={n} ({samples} states) ...", flush=True)
        planar, absolute = simulate_joint_distributions(
            n=n, samples=samples, seed=seed, batch_size=batch_size
        )
        output = RESULTS_DIR / f"comparison_n{n}.npz"
        np.savez_compressed(
            output,
            planar=planar,
            absolute=absolute,
            n=n,
            seed=seed,
        )
        print("Saved", output)


def simulate_publication_validation(
    samples: int, n10_samples: int, batch_size: int
) -> None:
    """Generate every new fixed-seed validation dataset used in v6."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cases = (
        (6, 2, (1, 2, 3), samples, 4606, "validation_general_p2_n6"),
        (4, 3, (1, 2), samples, 4704, "validation_general_p3_n4"),
        (9, 2, (4,), samples, 4909, "validation_balanced_n9"),
        (10, 2, (5,), n10_samples, 4910, "validation_balanced_n10"),
    )
    for n, local_dimension, block_sizes, count, seed, stem in cases:
        print(
            f"Sampling n={n}, p={local_dimension}, "
            f"k={block_sizes} ({count} states) ...",
            flush=True,
        )
        distributions = simulate_planar_distributions(
            n=n,
            local_dimension=local_dimension,
            block_sizes=block_sizes,
            samples=count,
            seed=seed,
            batch_size=batch_size,
        )
        output = RESULTS_DIR / f"{stem}.npz"
        payload: dict[str, np.ndarray | int] = {
            f"k{size}": values for size, values in distributions.items()
        }
        payload.update(
            {
                "n": n,
                "local_dimension": local_dimension,
                "seed": seed,
                "samples": count,
            }
        )
        np.savez_compressed(output, **payload)
        print("Saved", output, flush=True)


def _save_figure(figure: plt.Figure, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_DIR / f"{stem}.pdf", bbox_inches="tight")
    figure.savefig(FIGURE_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_standardized_distributions() -> None:
    # Four representative sizes keep the finite-size shape comparison legible.
    ns = [4, 5, 8, 10]
    colors = plt.cm.viridis(np.linspace(0.05, 0.9, len(ns)))
    figure, axis = plt.subplots(figsize=(7.1, 4.3))
    bins = np.linspace(-3.5, 5.0, 90)
    for n, color in zip(ns, colors):
        values = load_manuscript_planar(n)
        standardized = (values - exact_planar_mean(n)) / math.sqrt(
            exact_planar_variance(n)
        )
        axis.hist(
            standardized,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.35,
            color=color,
            label=rf"$n={n}$",
        )
    grid = np.linspace(bins[0], bins[-1], 600)
    axis.plot(grid, norm.pdf(grid), "k--", linewidth=1.6, label="standard normal")
    axis.set_xlabel(r"standardized planar purity $(\pi_{\rm P}-\mu_n)/\sigma_n$")
    axis.set_ylabel("density")
    axis.set_xlim(bins[0], bins[-1])
    axis.legend(ncol=2, frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    _save_figure(figure, "planar_standardized_distributions")


def plot_moment_validation() -> None:
    ns = np.arange(4, 11)
    statistics = [
        sample_statistics(load_manuscript_planar(int(n))) for n in ns
    ]
    sample_mean = np.array([row["mean"] for row in statistics])
    sample_variance = np.array([row["variance"] for row in statistics])
    sample_skew = np.array([row["skewness"] for row in statistics])
    sample_mean_se = np.array([row["mean_se"] for row in statistics])
    sample_variance_se = np.array([row["variance_se"] for row in statistics])
    sample_skew_se = np.array([row["skewness_se"] for row in statistics])
    theory_mean = np.array([exact_planar_mean(int(n)) for n in ns])
    theory_variance = np.array([exact_planar_variance(int(n)) for n in ns])
    theory_skew = np.array([exact_planar_skewness(int(n)) for n in ns])

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(10.2, 5.35),
        gridspec_kw={"height_ratios": (1.15, 0.85)},
        sharex="col",
    )
    panels = (
        (sample_mean, sample_mean_se, theory_mean, "mean", False),
        (sample_variance, sample_variance_se, theory_variance, "variance", True),
        (sample_skew, sample_skew_se, theory_skew, "skewness", False),
    )
    for column, (observed, uncertainty, exact, label, log_scale) in enumerate(panels):
        axis = axes[0, column]
        axis.plot(ns, exact, color="#1f77b4", linewidth=1.8, label="exact")
        axis.errorbar(
            ns,
            observed,
            yerr=uncertainty,
            fmt="o",
            color="#d95f02",
            markersize=4.8,
            capsize=2.4,
            linewidth=1.0,
            zorder=3,
            label=r"samples ($1$ s.e.)",
        )
        axis.set_ylabel(label)
        if log_scale:
            axis.set_yscale("log")
        axis.spines[["top", "right"]].set_visible(False)

        residual_axis = axes[1, column]
        residuals = (observed - exact) / uncertainty
        residual_axis.axhspan(-2, 2, color="0.92", zorder=0)
        residual_axis.axhspan(-1, 1, color="0.84", zorder=0)
        residual_axis.axhline(0, color="0.35", linewidth=1.0, zorder=1)
        residual_axis.plot(
            ns,
            residuals,
            linestyle="none",
            marker="o",
            color="#d95f02",
            markersize=4.8,
            zorder=2,
        )
        residual_axis.set_xlabel("number of qubits $n$")
        residual_axis.set_ylabel(r"residual / s.e.")
        residual_axis.set_xticks(ns)
        residual_axis.set_ylim(-2.55, 2.55)
        residual_axis.set_yticks((-2, -1, 0, 1, 2))
        residual_axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False)
    figure.tight_layout()
    _save_figure(figure, "moment_validation")


def plot_planar_absolute_comparison() -> None:
    datasets: list[tuple[int, np.ndarray, np.ndarray]] = []
    for n in (4, 8):
        loaded = np.load(RESULTS_DIR / f"comparison_n{n}.npz")
        datasets.append((n, loaded["planar"], loaded["absolute"]))
    datasets.append((10, load_saved_planar(10), load_saved_absolute(10)))

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(10.2, 5.4),
        gridspec_kw={"height_ratios": (1.0, 0.78)},
    )
    for column, (n, planar, absolute) in enumerate(datasets):
        axis = axes[0, column]
        lower = min(planar.min(), absolute.min())
        upper = max(planar.max(), absolute.max())
        bins = np.linspace(lower, upper, 75)
        axis.hist(
            planar,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.6,
            color="#0072B2",
            label="planar",
        )
        axis.hist(
            absolute,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.6,
            color="#D55E00",
            label="absolute",
        )
        shared_mean = exact_planar_mean(n)
        variance_ratio = exact_planar_variance(n) / exact_absolute_variance_even(n)
        axis.axvline(
            shared_mean,
            color="0.25",
            linestyle=":",
            linewidth=1.25,
            label="shared exact mean" if n == 4 else None,
        )
        axis.set_title(
            rf"$n={n}$" "\n"
            rf"$\mathrm{{Var}}_{{\rm P}}/\mathrm{{Var}}_{{\rm A}}={variance_ratio:.2f}$",
            fontsize=10,
        )
        axis.set_xlabel("purity")
        axis.spines[["top", "right"]].set_visible(False)
        difference = planar - absolute
        difference_axis = axes[1, column]
        bound = float(np.max(np.abs(difference)))
        difference_bins = np.linspace(-bound, bound, 75)
        difference_axis.hist(
            difference,
            bins=difference_bins,
            density=True,
            histtype="stepfilled",
            alpha=0.35,
            linewidth=1.2,
            color="#6A3D9A",
        )
        difference_axis.axvline(0.0, color="0.25", linestyle=":", linewidth=1.2)
        difference_axis.set_xlabel(r"paired difference $\pi_{\rm P}-\pi_{\rm A}$")
        difference_axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("density")
    axes[1, 0].set_ylabel("density")
    axes[0, 0].legend(frameon=False)
    figure.tight_layout()
    _save_figure(figure, "planar_vs_absolute")


def plot_geometry_and_variance_ratio() -> None:
    """Show geometric purity correlations and planar/absolute variance ratios."""

    figure, axes = plt.subplots(1, 2, figsize=(7.1, 3.15))

    n = 10
    for block_size, color, marker in (
        (1, "#0072B2", "o"),
        (3, "#E69F00", "^"),
        (5, "#009E73", "s"),
    ):
        # The balanced k=5 endpoint is the complementary interval and hence
        # has correlation one; k=1 and 3 show the nonbalanced kernels.
        displacements = np.arange(n // 2 + 1)
        single_variance = exact_planar_pair_covariance(
            n, 0, block_size=block_size
        )
        correlations = np.array(
            [
                exact_planar_pair_covariance(
                    n, int(displacement), block_size=block_size
                )
                / single_variance
                for displacement in displacements
            ]
        )
        axes[0].plot(
            displacements,
            correlations,
            marker=marker,
            color=color,
            linewidth=1.5,
            markersize=4.5,
            label=rf"$k={block_size}$",
        )
    axes[0].set_xlabel("cyclic displacement $d$")
    axes[0].set_ylabel(r"correlation $\rho_d$")
    axes[0].set_xticks(np.arange(6))
    axes[0].set_ylim(0, 1.06)
    axes[0].legend(frameon=False)

    even_ns = np.arange(4, 26, 2)
    ratios = np.array(
        [
            exact_planar_variance(int(n)) / exact_absolute_variance_even(int(n))
            for n in even_ns
        ]
    )
    axes[1].plot(
        even_ns,
        ratios,
        marker="o",
        color="#009E73",
        linewidth=1.6,
        markersize=4.3,
    )
    axes[1].set_xlabel("number of qubits $n$")
    axes[1].set_ylabel(r"$\operatorname{Var}(\pi_{\rm P})/\operatorname{Var}(\pi_{\rm A})$")
    axes[1].set_yscale("log")
    axes[1].set_xticks(np.arange(4, 26, 4))
    axes[1].text(
        0.96,
        0.06,
        r"asymptotically $\propto N^{0.415}/\log N$",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        color="0.25",
    )

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    _save_figure(figure, "geometry_and_variance_ratio")


def plot_partition_diagram() -> None:
    figure, axis = plt.subplots(figsize=(5.2, 3.55), subplot_kw={"aspect": "equal"})
    n = 12
    k = 4
    angles = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, n, endpoint=False)
    xy = np.column_stack((np.cos(angles), np.sin(angles)))
    in_a = np.arange(k)
    in_b = np.arange(k, n)
    axis.scatter(xy[in_a, 0], xy[in_a, 1], s=72, color="#0072B2", zorder=3)
    axis.scatter(xy[in_b, 0], xy[in_b, 1], s=72, color="#D55E00", zorder=3)
    circle = plt.Circle((0, 0), 1, fill=False, color="0.55", linewidth=1.1)
    axis.add_patch(circle)
    arc_angles = np.linspace(
        angles[0] + np.pi / n,
        angles[k - 1] - np.pi / n,
        120,
    )
    axis.plot(
        1.11 * np.cos(arc_angles),
        1.11 * np.sin(arc_angles),
        color="#0072B2",
        linewidth=3.0,
        solid_capstyle="round",
    )
    second_start = 2
    second_arc_angles = np.linspace(
        angles[second_start] + np.pi / n,
        angles[second_start + k - 1] - np.pi / n,
        120,
    )
    axis.plot(
        1.22 * np.cos(second_arc_angles),
        1.22 * np.sin(second_arc_angles),
        color="#009E73",
        linewidth=2.2,
        linestyle="--",
        solid_capstyle="round",
    )
    callout_box = {
        "boxstyle": "round,pad=0.18",
        "facecolor": "white",
        "edgecolor": "none",
        "alpha": 0.92,
    }
    axis.annotate(
        r"start $s$",
        xy=xy[0],
        xytext=(-0.62, 1.36),
        fontsize=9.5,
        ha="center",
        va="center",
        bbox=callout_box,
        arrowprops={"arrowstyle": "->", "color": "0.25", "lw": 1.0},
    )
    axis.annotate(
        r"start $t$",
        xy=xy[second_start],
        xytext=(1.46, 0.79),
        fontsize=9.5,
        ha="center",
        va="center",
        color="#00825C",
        bbox=callout_box,
        arrowprops={"arrowstyle": "->", "color": "#00825C", "lw": 1.0},
    )
    axis.annotate(
        r"$s\rightarrow t$",
        xy=(1.32 * np.cos(angles[2]), 1.32 * np.sin(angles[2])),
        xytext=(1.32 * np.cos(angles[0]), 1.32 * np.sin(angles[0])),
        fontsize=8.8,
        ha="center",
        va="bottom",
        bbox=callout_box,
        arrowprops={
            "arrowstyle": "->",
            "color": "0.25",
            "lw": 1.0,
            "connectionstyle": "arc3,rad=-0.20",
        },
    )
    axis.annotate(
        r"$A_t^{(k)}$",
        xy=(1.22 * np.cos(np.mean(second_arc_angles)),
            1.22 * np.sin(np.mean(second_arc_angles))),
        xytext=(1.53, -0.25),
        color="#00825C",
        fontsize=10.0,
        ha="center",
        va="center",
        bbox=callout_box,
        arrowprops={"arrowstyle": "-", "color": "#00825C", "lw": 0.9},
    )
    axis.text(
        0.10,
        -1.39,
        r"$q=|A_s^{(k)}\cap A_t^{(k)}|=2$",
        color="#00825C",
        fontsize=8.8,
        ha="center",
        va="center",
        bbox=callout_box,
    )
    axis.text(
        0.42,
        0.34,
        r"$A_s^{(k)}$" "\n" r"($k$ sites)",
        color="#0072B2",
        fontsize=10.0,
        ha="center",
        va="center",
    )
    axis.text(
        -0.40,
        -0.22,
        r"$\overline{A_s^{(k)}}$" "\n" r"($n-k$ sites)",
        color="#D55E00",
        fontsize=9.2,
        ha="center",
        va="center",
    )
    axis.set_xlim(-1.43, 1.76)
    axis.set_ylim(-1.53, 1.53)
    axis.axis("off")
    figure.tight_layout()
    _save_figure(figure, "planar_partition")


def generate_figures() -> None:
    plot_partition_diagram()
    plot_standardized_distributions()
    plot_moment_validation()
    plot_planar_absolute_comparison()
    plot_geometry_and_variance_ratio()
    print("Saved figures in", FIGURE_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("summarize", help="write sample and exact-theory CSV summaries")
    simulation = subparsers.add_parser(
        "simulate-comparisons", help="regenerate paired planar/absolute samples"
    )
    simulation.add_argument("--samples", type=int, default=40_000)
    simulation.add_argument("--batch-size", type=int, default=256)
    validation = subparsers.add_parser(
        "simulate-validation",
        help="generate fixed-seed nonbalanced, qudit, n=9, and n=10 samples",
    )
    validation.add_argument("--samples", type=int, default=40_000)
    validation.add_argument("--n10-samples", type=int, default=120_000)
    validation.add_argument("--batch-size", type=int, default=256)
    subparsers.add_parser("figures", help="generate all manuscript figures")
    args = parser.parse_args()

    if args.command == "summarize":
        write_summaries()
    elif args.command == "simulate-comparisons":
        simulate_comparisons(samples=args.samples, batch_size=args.batch_size)
    elif args.command == "simulate-validation":
        simulate_publication_validation(
            samples=args.samples,
            n10_samples=args.n10_samples,
            batch_size=args.batch_size,
        )
    elif args.command == "figures":
        generate_figures()


if __name__ == "__main__":
    main()
