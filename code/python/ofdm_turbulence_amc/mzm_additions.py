"""
MZM optical OFDM additions — Hermitian symmetric real-valued OFDM + MZM nonlinearity.
Append to simulation.py or import separately.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


# ============================
# MZM Configuration
# ============================
@dataclass(frozen=True)
class MZMConfig:
    """Configuration for the MZM optical OFDM simulation chain.

    The chain:
      Hermitian IFFT (real-valued) → Add CP → MZM sin transfer → Gamma-Gamma → AWGN
    """
    sequence_length: int = 8192
    n_subcarriers: int = 64         # IFFT size (must be even for Hermitian symmetry)
    cp_length: int = 8
    # MZM parameters
    alpha: float = 0.85             # Drive swing ratio (0 < alpha <= 1)
    v_pi: float = 5.0               # Half-wave voltage (V)
    # Turbulence
    turbulence_coherence_symbols: int = 1_000_000
    # Phase/CFO/STO disabled for intensity-modulated direct-detection (IM/DD)
    # (IM/DD is inherently real-valued and phase-insensitive)


# ============================
# Hermitian Symmetric OFDM
# ============================
def hermitian_ofdm_time_signal(
    symbols: np.ndarray,        # (n_symbols, n_data) QAM symbols
    n_fft: int,                 # IFFT size (even)
    rng: np.random.Generator,
) -> np.ndarray:
    """Pack QAM symbols with Hermitian symmetry → IFFT → real-valued time signal.

    Uses DC = 0, Nyquist = 0.
    Data subcarriers: indices 1 .. n_fft//2 - 1  (yielding n_fft//2 - 1 data carriers)
    Conjugate symmetric: X[N-k] = conj(X[k])
    """
    n_data = n_fft // 2 - 1
    n_symbols = symbols.shape[0]

    freq_grid = np.zeros((n_symbols, n_fft), dtype=np.complex64)
    freq_grid[:, 1:n_data + 1] = symbols

    # Hermitian symmetric conjugate pairs
    for k in range(1, n_data + 1):
        freq_grid[:, n_fft - k] = np.conj(freq_grid[:, k])

    # IFFT → real-valued output (within numerical precision)
    time_signal = np.fft.ifft(freq_grid, axis=1, norm="ortho")
    # Drop near-zero imaginary parts from numerical rounding
    time_signal = time_signal.real.astype(np.float32)
    # Normalize to unit variance
    time_signal /= (np.std(time_signal, axis=1, keepdims=True) + 1e-12)
    return time_signal


# ============================
# MZM Nonlinear Transfer
# ============================
def mzm_modulate(
    x_t: np.ndarray,        # (n_symbols, n_fft) real-valued OFDM time signal per symbol
    alpha: float,
    v_pi: float,
) -> np.ndarray:
    """Apply MZM sinusoidal transfer curve.

    P_out = P_in/2 * [1 + sin(pi * V(t) / V_pi)]

    where V(t) = alpha * (V_pi / 2) * x_t / max(|x_t|)
    """
    peak = float(np.max(np.abs(x_t)))
    if peak < 1e-12:
        peak = 1.0
    # Drive voltage
    v_t = alpha * (v_pi / 2.0) * (x_t / peak)
    # MZM output optical power
    p_out = 0.5 * (1.0 + np.sin(np.pi * v_t / v_pi))
    return p_out.astype(np.float32)   # non-negative real-valued intensity


# ============================
# Full Optical OFDM Generator
# ============================
def generate_optical_ofdm_with_metadata(
    modulation_index: int,
    snr_db: float,
    turbulence: str,
    sample_index: int,
    base_seed: int,
    config: MZMConfig,
    constellation: np.ndarray,   # pre-loaded QAM constellation from CONSTELLATIONS
) -> tuple[np.ndarray, dict]:
    """Generate a real-valued optical OFDM sequence with MZM + turbulence + noise.

    Returns (sequence_1d, metadata).
    sequence_1d: shape (sequence_length,) float32 — the received photocurrent.
    """
    try:
        from .simulation import _seed_from, gamma_gamma_irradiance, MODULATION_ORDERS
    except ImportError:
        from simulation import _seed_from, gamma_gamma_irradiance, MODULATION_ORDERS

    seed = _seed_from(
        base_seed, modulation_index, snr_db, turbulence, sample_index, config
    )
    rng = np.random.default_rng(seed)

    order = MODULATION_ORDERS[modulation_index]
    symbol_length = config.n_subcarriers + config.cp_length
    n_data = config.n_subcarriers // 2 - 1
    n_symbols = int(np.ceil(config.sequence_length / symbol_length)) + 1

    # Generate QAM symbols
    indices = rng.integers(0, constellation.size, size=(n_symbols, n_data))
    frequency_symbols = constellation[indices]

    # Hermitian IFFT → real-valued time domain
    time_signal = hermitian_ofdm_time_signal(
        frequency_symbols, config.n_subcarriers, rng
    )  # (n_symbols, n_fft)

    # Add CP
    with_cp = np.concatenate(
        [time_signal[:, -config.cp_length:], time_signal], axis=1
    )  # (n_symbols, symbol_length)
    transmitted = with_cp.reshape(-1)  # 1D real-valued

    # MZM nonlinear transfer
    optical_power = mzm_modulate(transmitted, config.alpha, config.v_pi)
    # optical_power is non-negative, represents instantaneous optical intensity

    # Gamma-Gamma turbulence (multiplicative on intensity)
    coherence = min(max(1, config.turbulence_coherence_symbols), n_symbols)
    n_fades = int(np.ceil(n_symbols / coherence))
    irradiance = gamma_gamma_irradiance(rng, turbulence, n_fades)
    intensity = np.empty(n_symbols, dtype=np.float32)
    for fade_index, value in enumerate(irradiance):
        start = fade_index * coherence
        end = min(start + coherence, n_symbols)
        intensity[start:end] = value

    # Apply turbulence to each symbol's intensity
    optical_per_symbol = with_cp * intensity[:, None]
    faded_optical = optical_per_symbol.reshape(-1)

    # Truncate to requested sequence length
    faded_optical = faded_optical[:config.sequence_length]

    # Add AWGN (SNR relative to mean optical power)
    mean_signal_power = float(np.mean(faded_optical ** 2))
    noise_power = mean_signal_power / (10.0 ** (snr_db / 10.0))
    noise = np.sqrt(noise_power) * rng.standard_normal(faded_optical.size)

    received = faded_optical + noise
    # Ensure non-negative (physical photocurrent)
    received = np.maximum(received, 0.0).astype(np.float32)

    metadata = {
        "modulation_index": modulation_index,
        "snr_db": snr_db,
        "turbulence": turbulence,
        "alpha": config.alpha,
        "n_subcarriers": config.n_subcarriers,
        "cp_length": config.cp_length,
        "noise_power": float(noise_power),
        "mean_signal_power": float(mean_signal_power),
    }
    return received, metadata


# ============================
# Optical OFDM Dataset
# ============================



class MZMOFDMDataset(torch.utils.data.Dataset):
    """Dataset for MZM optical OFDM with configurable alpha and turbulence."""

    def __init__(
        self,
        modulations: tuple[str, ...],
        modulation_orders: tuple[int, ...],
        constellations: dict,
        alphas: list[float],
        turbulence_levels: list[str],
        snrs: list[float],
        examples_per_condition: int,
        seed: int,
        config_template: MZMConfig,
        cache: bool = False,
    ):
        self.modulations = modulations
        self.modulation_orders = modulation_orders
        self.constellations = constellations
        self.alphas = alphas
        self.turbulence_levels = turbulence_levels
        self.snrs = snrs
        self.examples_per_condition = examples_per_condition
        self.seed = seed
        self.config_template = config_template
        self.cache = cache
        self._cached_data = None

        self.records = [
            (mod_idx, snr, turb, alpha, sample_idx)
            for turb in turbulence_levels
            for snr in snrs
            for alpha in alphas
            for mod_idx in range(len(modulations))
            for sample_idx in range(examples_per_condition)
        ]

        if cache:
            self._cached_data = [self._generate(idx) for idx in range(len(self.records))]

    def _generate(self, index: int):
        mod_idx, snr, turb, alpha, sample_idx = self.records[index]
        config = MZMConfig(
            sequence_length=self.config_template.sequence_length,
            n_subcarriers=self.config_template.n_subcarriers,
            cp_length=self.config_template.cp_length,
            alpha=alpha,
            v_pi=self.config_template.v_pi,
            turbulence_coherence_symbols=self.config_template.turbulence_coherence_symbols,
        )
        sequence, _ = generate_optical_ofdm_with_metadata(
            mod_idx, snr, turb, sample_idx, self.seed, config,
            self.constellations[self.modulation_orders[mod_idx]],
        )
        return sequence

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        if self._cached_data is not None:
            sequence = self._cached_data[index]
        else:
            sequence = self._generate(index)
        mod_idx = self.records[index][0]
        snr = self.records[index][1]
        turb = self.records[index][2]
        # Shape: (1, seq_len) — single-channel real-valued intensity
        sequence_tensor = torch.from_numpy(sequence).unsqueeze(0)
        return (
            sequence_tensor,
            torch.tensor(mod_idx, dtype=torch.long),
            torch.tensor(float(snr), dtype=torch.float32),
            turb,
        )
