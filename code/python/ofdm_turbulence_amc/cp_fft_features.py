"""Blind CP synchronization and noise-aware FFT constellation features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

try:
    from .simulation import CONSTELLATIONS, MODULATION_ORDERS, SimulationConfig
except ImportError:
    from simulation import CONSTELLATIONS, MODULATION_ORDERS, SimulationConfig


@dataclass(frozen=True)
class SynchronizationResult:
    boundary_offset: int
    cfo: float
    metric: float


def _as_complex(sequence: np.ndarray) -> np.ndarray:
    return sequence[0].astype(np.float64) + 1j * sequence[1].astype(np.float64)


def estimate_cp_synchronization(
    sequence: np.ndarray,
    config: SimulationConfig,
) -> SynchronizationResult:
    """Estimate periodic OFDM boundary and CFO from CP/tail correlation."""
    x = _as_complex(sequence)
    n_fft = config.n_subcarriers
    cp = config.cp_length
    symbol_length = n_fft + cp
    best_offset = 0
    best_metric = -1.0
    best_correlation = 0.0j

    for offset in range(symbol_length):
        starts = np.arange(
            offset,
            x.size - symbol_length + 1,
            symbol_length,
            dtype=np.int64,
        )
        if starts.size == 0:
            continue
        cp_indices = (starts[:, None] + np.arange(cp)).reshape(-1)
        tail_indices = cp_indices + n_fft
        first = x[cp_indices]
        second = x[tail_indices]
        correlation = np.vdot(first, second)
        denominator = np.sqrt(
            np.vdot(first, first).real * np.vdot(second, second).real + 1e-12
        )
        metric = float(np.abs(correlation) / denominator)
        if metric > best_metric:
            best_metric = metric
            best_offset = offset
            best_correlation = correlation

    # vdot(CP, tail) has phase +2*pi*CFO because the copies are N samples apart.
    cfo = float(np.angle(best_correlation) / (2.0 * np.pi))
    return SynchronizationResult(best_offset, cfo, best_metric)


def correct_cfo(sequence: np.ndarray, cfo: float, n_fft: int) -> np.ndarray:
    x = _as_complex(sequence)
    sample_index = np.arange(x.size, dtype=np.float64)
    return x * np.exp(-1j * 2.0 * np.pi * cfo * sample_index / n_fft)


def recover_subcarriers(
    sequence: np.ndarray,
    config: SimulationConfig,
    boundary_offset: int,
    cfo: float,
    merge_cp: bool,
) -> tuple[np.ndarray, float, int]:
    """CFO-correct, segment symbols, optionally merge CP, and apply FFT."""
    x = correct_cfo(sequence, cfo, config.n_subcarriers)
    n_fft = config.n_subcarriers
    cp = config.cp_length
    symbol_length = n_fft + cp
    starts = np.arange(
        boundary_offset,
        x.size - symbol_length + 1,
        symbol_length,
        dtype=np.int64,
    )
    if starts.size == 0:
        raise ValueError("No complete OFDM symbol remains after synchronization.")

    blocks = np.stack([x[start : start + symbol_length] for start in starts])
    prefixes = blocks[:, :cp]
    useful = blocks[:, cp:].copy()
    tails = useful[:, n_fft - cp :]
    residual = tails - prefixes
    noise_variance = float(np.mean(np.abs(residual) ** 2) / 2.0)
    if merge_cp:
        useful[:, n_fft - cp :] = 0.5 * (tails + prefixes)
    subcarriers = np.fft.fft(useful, axis=1, norm="ortho").reshape(-1)
    return subcarriers, noise_variance, int(starts.size)


def _radial_likelihood_features(
    normalized_amplitude: np.ndarray,
    normalized_noise_variance: float,
) -> tuple[list[float], list[str]]:
    values: list[float] = []
    names: list[str] = []
    sample = normalized_amplitude
    if sample.size > 384:
        indices = np.linspace(0, sample.size - 1, 384, dtype=np.int64)
        sample = np.sort(sample)[indices]
    scales = np.linspace(0.78, 1.22, 13)
    noise = float(np.clip(normalized_noise_variance, 2e-3, 8.0))

    scores = []
    distances = []
    for order in MODULATION_ORDERS:
        raw_radii = np.abs(CONSTELLATIONS[order]).astype(np.float64)
        radii, counts = np.unique(np.round(raw_radii, 7), return_counts=True)
        log_weights = np.log(counts.astype(np.float64) / counts.sum())
        centers = scales[:, None] * radii[None, :]
        squared_distance = (
            sample[:, None, None] - centers[None, :, :]
        ) ** 2
        log_density = logsumexp(
            -squared_distance / noise + log_weights[None, None, :],
            axis=2,
        )
        scale_scores = np.mean(log_density, axis=0) - 0.5 * np.log(noise)
        best_score = float(np.max(scale_scores))
        best_distance = float(
            np.min(np.mean(np.min(squared_distance, axis=2), axis=0))
        )
        scores.append(best_score)
        distances.append(best_distance)
        names.extend(
            [
                f"radial_log_likelihood_{order}",
                f"radial_nearest_distance_{order}",
            ]
        )
        values.extend([best_score, best_distance])

    sorted_scores = np.sort(np.asarray(scores))
    values.append(float(sorted_scores[-1] - sorted_scores[-2]))
    names.append("radial_likelihood_top_margin")
    score_map = dict(zip(MODULATION_ORDERS, scores))
    distance_map = dict(zip(MODULATION_ORDERS, distances))
    for first, second in ((32, 128), (64, 256)):
        values.extend(
            [
                float(score_map[second] - score_map[first]),
                float(distance_map[second] - distance_map[first]),
            ]
        )
        names.extend(
            [
                f"radial_log_likelihood_delta_{second}_{first}",
                f"radial_distance_delta_{second}_{first}",
            ]
        )
    return values, names


def extract_constellation_features(
    subcarriers: np.ndarray,
    noise_variance: float,
    sync_metric: float,
    cfo_estimate: float,
    symbol_count: int,
) -> tuple[np.ndarray, list[str]]:
    """Build fixed-dimensional constellation distribution and likelihood features."""
    eps = 1e-12
    points = subcarriers - np.mean(subcarriers)
    received_power = float(np.mean(np.abs(points) ** 2))
    signal_power = max(received_power - noise_variance, received_power * 0.05, eps)
    normalized = points / np.sqrt(signal_power)
    amplitude = np.abs(normalized)
    normalized_noise = float(np.clip(noise_variance / signal_power, 1e-6, 1e3))

    values: list[float] = []
    names: list[str] = []

    def add(name: str, value: float) -> None:
        names.append(name)
        values.append(
            float(np.nan_to_num(value, nan=0.0, posinf=1e6, neginf=-1e6))
        )

    histogram, _ = np.histogram(amplitude, bins=64, range=(0.0, 2.5))
    histogram = histogram.astype(np.float64) / max(amplitude.size, 1)
    for index, value in enumerate(histogram):
        add(f"amplitude_histogram_{index:02d}", value)
    add("amplitude_histogram_overflow", np.mean(amplitude >= 2.5))

    thresholds = np.linspace(0.1, 2.4, 32)
    for index, threshold in enumerate(thresholds):
        add(f"amplitude_cdf_{index:02d}", np.mean(amplitude <= threshold))

    for quantile in (
        0.01, 0.05, 0.10, 0.20, 0.25, 0.40, 0.50, 0.60, 0.75,
        0.80, 0.85, 0.90, 0.95, 0.97, 0.99,
    ):
        add(f"amplitude_quantile_{quantile:.2f}", np.quantile(amplitude, quantile))

    for threshold in (0.4, 0.7, 1.0, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0, 2.2):
        add(f"ring_occupancy_below_{threshold:.1f}", np.mean(amplitude <= threshold))

    fine_histogram, _ = np.histogram(amplitude, bins=96, range=(0.0, 2.4))
    probabilities = fine_histogram.astype(np.float64)
    probabilities /= max(probabilities.sum(), 1.0)
    nonzero = probabilities[probabilities > 0]
    add("ring_histogram_entropy", -np.sum(nonzero * np.log(nonzero)))
    add("outer_ring_occupancy_1p4", np.mean(amplitude >= 1.4))
    add("outer_ring_occupancy_1p6", np.mean(amplitude >= 1.6))
    add("outer_ring_occupancy_1p8", np.mean(amplitude >= 1.8))
    add("outer_tail_mean", np.mean(amplitude[amplitude >= np.quantile(amplitude, 0.9)]))

    m2 = float(np.mean(np.abs(normalized) ** 2)) + eps
    m4 = float(np.mean(np.abs(normalized) ** 4))
    m6 = float(np.mean(np.abs(normalized) ** 6))
    ex2 = np.mean(normalized**2)
    ex4 = np.mean(normalized**4)
    ex6 = np.mean(normalized**6)
    c40 = ex4 - 3.0 * ex2**2
    c42 = m4 - np.abs(ex2) ** 2 - 2.0 * m2**2
    c60 = ex6 - 15.0 * ex4 * ex2 + 30.0 * ex2**3
    add("normalized_moment_4", m4 / m2**2)
    add("normalized_moment_6", m6 / m2**3)
    add("normalized_abs_c40", np.abs(c40) / m2**2)
    add("normalized_abs_c42", np.abs(c42) / m2**2)
    add("normalized_abs_c60", np.abs(c60) / m2**3)
    add("phase_circular_abs_4", np.abs(np.mean(np.exp(4j * np.angle(normalized)))))
    add("phase_circular_abs_8", np.abs(np.mean(np.exp(8j * np.angle(normalized)))))

    likelihood_values, likelihood_names = _radial_likelihood_features(
        amplitude, normalized_noise
    )
    values.extend(likelihood_values)
    names.extend(likelihood_names)

    add("log_normalized_noise_variance", np.log10(normalized_noise + eps))
    add("cp_sync_metric", sync_metric)
    add("abs_cfo_estimate", abs(cfo_estimate))
    add("ofdm_symbol_count", symbol_count)
    return np.asarray(values, dtype=np.float32), names


def synchronization_errors(
    estimate: SynchronizationResult,
    true_boundary: int,
    true_cfo: float,
    symbol_length: int,
) -> tuple[int, float]:
    raw = abs(int(estimate.boundary_offset) - int(true_boundary))
    boundary_error = min(raw, symbol_length - raw)
    cfo_error = abs(float(estimate.cfo) - float(true_cfo))
    return boundary_error, cfo_error
