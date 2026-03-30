from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray


def _first_primes(count: int) -> list[int]:
    primes: list[int] = []
    candidate = 2
    while len(primes) < count:
        is_prime = True
        limit = int(math.sqrt(candidate))
        for prime in primes:
            if prime > limit:
                break
            if candidate % prime == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(candidate)
        candidate += 1
    return primes


def _radical_inverse(index: int, base: int) -> float:
    result = 0.0
    factor = 1.0 / base
    while index > 0:
        index, remainder = divmod(index, base)
        result += remainder * factor
        factor /= base
    return result


def halton_uniforms(
    num_samples: int,
    dimension: int,
    *,
    seed: int | None = None,
    skip: int = 32,
) -> NDArray[np.float64]:
    if num_samples < 1:
        raise ValueError("num_samples must be positive.")
    if dimension < 1:
        raise ValueError("dimension must be positive.")

    primes = _first_primes(dimension)
    sequence = np.empty((num_samples, dimension), dtype=np.float64)
    for dim, base in enumerate(primes):
        for sample in range(num_samples):
            sequence[sample, dim] = _radical_inverse(sample + skip + 1, base)

    if seed is not None:
        rng = np.random.default_rng(seed)
        shifts = rng.random(dimension, dtype=np.float64)
        sequence = np.mod(sequence + shifts, 1.0)

    return np.clip(sequence, 1e-12, 1.0 - 1e-12)


def inverse_normal_cdf(values: NDArray[np.float64]) -> NDArray[np.float64]:
    a = np.array(
        [
            -39.69683028665376,
            220.9460984245205,
            -275.9285104469687,
            138.3577518672690,
            -30.66479806614716,
            2.506628277459239,
        ],
        dtype=np.float64,
    )
    b = np.array(
        [
            -54.47609879822406,
            161.5858368580409,
            -155.6989798598866,
            66.80131188771972,
            -13.28068155288572,
        ],
        dtype=np.float64,
    )
    c = np.array(
        [
            -0.007784894002430293,
            -0.3223964580411365,
            -2.400758277161838,
            -2.549732539343734,
            4.374664141464968,
            2.938163982698783,
        ],
        dtype=np.float64,
    )
    d = np.array(
        [
            0.007784695709041462,
            0.3224671290700398,
            2.445134137142996,
            3.754408661907416,
        ],
        dtype=np.float64,
    )
    lower = 0.02425
    upper = 1.0 - lower

    clipped = np.clip(values, 1e-12, 1.0 - 1e-12)
    result = np.empty_like(clipped)

    lower_mask = clipped < lower
    upper_mask = clipped > upper
    center_mask = ~(lower_mask | upper_mask)

    if np.any(lower_mask):
        q = np.sqrt(-2.0 * np.log(clipped[lower_mask]))
        numerator = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        denominator = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        result[lower_mask] = numerator / denominator

    if np.any(center_mask):
        q = clipped[center_mask] - 0.5
        r = q * q
        numerator = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        denominator = (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        result[center_mask] = numerator / denominator

    if np.any(upper_mask):
        q = np.sqrt(-2.0 * np.log(1.0 - clipped[upper_mask]))
        numerator = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        denominator = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        result[upper_mask] = -(numerator / denominator)

    return result
