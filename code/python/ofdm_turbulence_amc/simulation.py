"""Deterministic or epoch-dynamic OFDM/QAM waveform simulation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


MODULATIONS = ("QPSK", "16QAM", "32QAM", "64QAM", "128QAM", "256QAM")
MODULATION_ORDERS = (4, 16, 32, 64, 128, 256)
TURBULENCE_LEVELS = ("weak", "moderate", "strong")
TRAINING_TURBULENCE_LEVELS = ("none", "weak", "moderate", "strong")

# Representative Gamma-Gamma shape parameters. Smaller values imply stronger
# irradiance fluctuations. They are explicit experiment assumptions, not fitted
# values from the current optical link.
GAMMA_GAMMA = {
    "weak": (11.6, 10.1),
    "moderate": (4.0, 1.9),
    "strong": (2.1, 1.1),
}


@dataclass(frozen=True)
class SimulationConfig:
    sequence_length: int = 4096
    n_subcarriers: int = 64
    cp_length: int = 16
    cfo_max: float = 0.0
    sto_max: int = 0
    phase_max: float = np.pi / 2
    # Atmospheric turbulence is much slower than an OFDM symbol. A value larger
    # than the symbols in one sequence gives one block-fading coefficient.
    turbulence_coherence_symbols: int = 1_000_000


def _seed_from(*values: object) -> int:
    payload = "|".join(str(value) for value in values).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def qam_constellation(order: int) -> np.ndarray:
    """Return normalized rectangular QAM constellations, including 32/128-QAM."""
    if order == 4:
        i_levels = q_levels = np.asarray([-1, 1], dtype=np.float32)
    else:
        i_count = 2 ** int(np.ceil(np.log2(order) / 2))
        q_count = order // i_count
        i_levels = np.arange(-(i_count - 1), i_count, 2, dtype=np.float32)
        q_levels = np.arange(-(q_count - 1), q_count, 2, dtype=np.float32)
    constellation = (
        i_levels[:, None] + 1j * q_levels[None, :]
    ).reshape(-1).astype(np.complex64)
    constellation /= np.sqrt(np.mean(np.abs(constellation) ** 2))
    if constellation.size != order:
        raise RuntimeError(f"Failed to construct {order}-QAM")
    return constellation


CONSTELLATIONS = {order: qam_constellation(order) for order in MODULATION_ORDERS}


def gamma_gamma_irradiance(
    rng: np.random.Generator, level: str, size: int
) -> np.ndarray:
    if level == "none":
        return np.ones(size, dtype=np.float32)
    alpha, beta = GAMMA_GAMMA[level]
    large_scale = rng.gamma(alpha, 1.0 / alpha, size=size)
    small_scale = rng.gamma(beta, 1.0 / beta, size=size)
    return (large_scale * small_scale).astype(np.float32)


def generate_sequence_with_metadata(
    modulation_index: int,
    snr_db: float,
    turbulence: str,
    sample_index: int,
    base_seed: int,
    config: SimulationConfig,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Generate one IQ tensor and simulation-only metadata for oracle controls."""
    seed = _seed_from(
        base_seed, modulation_index, snr_db, turbulence, sample_index, config
    )
    rng = np.random.default_rng(seed)
    order = MODULATION_ORDERS[modulation_index]
    constellation = CONSTELLATIONS[order]
    symbol_length = config.n_subcarriers + config.cp_length
    n_symbols = int(
        np.ceil((config.sequence_length + config.sto_max) / symbol_length)
    ) + 1

    indices = rng.integers(
        0, constellation.size, size=(n_symbols, config.n_subcarriers)
    )
    frequency_symbols = constellation[indices]
    time_symbols = np.fft.ifft(frequency_symbols, axis=1, norm="ortho")
    with_cp = np.concatenate(
        [time_symbols[:, -config.cp_length :], time_symbols], axis=1
    )

    coherence = min(max(1, config.turbulence_coherence_symbols), n_symbols)
    n_fades = int(np.ceil(n_symbols / coherence))
    irradiance = gamma_gamma_irradiance(rng, turbulence, n_fades)
    amplitude = np.empty(n_symbols, dtype=np.float32)
    for fade_index, value in enumerate(np.sqrt(irradiance).astype(np.float32)):
        start = fade_index * coherence
        end = min(start + coherence, n_symbols)
        amplitude[start:end] = value
    transmitted = with_cp.reshape(-1)
    faded = with_cp * amplitude[:, None]
    sequence = faded.reshape(-1)

    # Random phase offset and normalized CFO, expressed in subcarrier spacings.
    phase = rng.uniform(-config.phase_max, config.phase_max)
    cfo = rng.uniform(-config.cfo_max, config.cfo_max)
    n = np.arange(sequence.size, dtype=np.float64)
    sequence *= np.exp(
        1j * (phase + 2 * np.pi * cfo * n / config.n_subcarriers)
    )

    sto = int(rng.integers(0, config.sto_max + 1))
    sequence = sequence[sto : sto + config.sequence_length]

    # SNR is referenced to the unfaded transmitted waveform. Deep fades
    # therefore reduce instantaneous received SNR instead of being cancelled by
    # re-scaling the noise to the faded signal.
    nominal_signal_power = float(np.mean(np.abs(transmitted) ** 2))
    noise_power = nominal_signal_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (
        rng.standard_normal(sequence.size) + 1j * rng.standard_normal(sequence.size)
    )
    received = sequence + noise
    iq = np.stack(
        [received.real.astype(np.float32), received.imag.astype(np.float32)]
    )
    metadata = {
        "cfo": float(cfo),
        "sto": int(sto),
        "phase": float(phase),
        "symbol_boundary_offset": int((-sto) % symbol_length),
        "noise_power": float(noise_power),
        "nominal_signal_power": float(nominal_signal_power),
    }
    return iq, metadata


def generate_sequence(
    modulation_index: int,
    snr_db: float,
    turbulence: str,
    sample_index: int,
    base_seed: int,
    config: SimulationConfig,
) -> np.ndarray:
    """Generate one independent 2xM real-valued IQ tensor."""
    sequence, _ = generate_sequence_with_metadata(
        modulation_index,
        snr_db,
        turbulence,
        sample_index,
        base_seed,
        config,
    )
    return sequence


class OFDMTurbulenceDataset(Dataset):
    def __init__(
        self,
        snrs: list[float],
        turbulence_levels: list[str],
        examples_per_condition: int,
        seed: int,
        config: SimulationConfig,
        cache: bool = False,
        dynamic_per_epoch: bool = False,
    ):
        self.snrs = snrs
        self.turbulence_levels = turbulence_levels
        self.examples_per_condition = examples_per_condition
        self.seed = seed
        self.config = config
        self.dynamic_per_epoch = dynamic_per_epoch
        self.epoch = 0
        self.records = [
            (modulation_index, snr, turbulence, sample_index)
            for turbulence in turbulence_levels
            for snr in snrs
            for modulation_index in range(len(MODULATIONS))
            for sample_index in range(examples_per_condition)
        ]
        self.cached = None
        if cache:
            self.cached = [
                generate_sequence(*record, seed, config) for record in self.records
            ]

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __getitem__(self, index: int):
        modulation_index, snr, turbulence, sample_index = self.records[index]
        active_seed = (
            self.seed + 1_000_003 * self.epoch if self.dynamic_per_epoch else self.seed
        )
        sequence = (
            self.cached[index]
            if self.cached is not None
            else generate_sequence(
                modulation_index,
                snr,
                turbulence,
                sample_index,
                active_seed,
                self.config,
            )
        )
        return (
            torch.from_numpy(sequence),
            torch.tensor(modulation_index, dtype=torch.long),
            torch.tensor(float(snr), dtype=torch.float32),
            turbulence,
        )


def iq_to_amp_phase(sequence: np.ndarray) -> np.ndarray:
    i = sequence[0]
    q = sequence[1]
    amplitude = np.sqrt(i * i + q * q)
    amplitude /= max(float(np.max(amplitude)), 1e-8)
    phase = np.angle(i + 1j * q) / np.pi
    return np.stack([amplitude.astype(np.float32), phase.astype(np.float32)])


def gramian_angular_fields(sequence: np.ndarray, image_size: int) -> np.ndarray:
    """Convert 2xM IQ sequence into GASF or GASF+GADF-ready channels."""
    if image_size < 16:
        raise ValueError("GAF image size must be at least 16.")
    channels = []
    source_length = sequence.shape[-1]
    if source_length % image_size == 0:
        reduced = sequence.reshape(2, image_size, source_length // image_size).mean(
            axis=2
        )
    else:
        old_axis = np.linspace(0.0, 1.0, source_length)
        new_axis = np.linspace(0.0, 1.0, image_size)
        reduced = np.stack(
            [np.interp(new_axis, old_axis, channel) for channel in sequence]
        )
    for channel in reduced:
        maximum = max(float(np.max(np.abs(channel))), 1e-8)
        scaled = np.clip(channel / maximum, -1.0, 1.0).astype(np.float32)
        sine = np.sqrt(np.maximum(0.0, 1.0 - scaled * scaled))
        # cos(phi_i + phi_j)
        gasf = np.outer(scaled, scaled) - np.outer(sine, sine)
        # sin(phi_i - phi_j)
        gadf = np.outer(sine, scaled) - np.outer(scaled, sine)
        channels.extend([gasf.astype(np.float32), gadf.astype(np.float32)])
    return np.stack(channels)


class GAFDataset(Dataset):
    """Representation-only wrapper; labels and simulated conditions are unchanged."""

    def __init__(self, base: Dataset, image_size: int, mode: str = "gasf_gadf"):
        self.base = base
        self.image_size = image_size
        self.mode = mode

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        sequence, label, snr, turbulence = self.base[index]
        image = gramian_angular_fields(sequence.numpy(), self.image_size)
        if self.mode == "gasf":
            image = image[[0, 2]]
        return torch.from_numpy(image), label, snr, turbulence


class AmplitudePhaseDataset(Dataset):
    """Return a 4xM tensor: I/Q plus amplitude/phase for dual-branch models."""

    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        sequence, label, snr, turbulence = self.base[index]
        amp_phase = iq_to_amp_phase(sequence.numpy())
        fused = np.concatenate([sequence.numpy(), amp_phase], axis=0)
        return torch.from_numpy(fused), label, snr, turbulence
