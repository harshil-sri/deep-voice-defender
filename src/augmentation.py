"""Augmentation strategies v2 — the paper's five ablation modes.

Modes (each run of the grid trains with exactly one):
  none     — no augmentation (raw training audio)
  noise    — additive Gaussian noise at a random SNR in [5, 20] dB
  resample — naive "codec" simulation: 16k -> 8k -> 16k decimation (v1's fake Opus;
             kept as the S1 head-to-head baseline against true Opus)
  opus     — TRUE Opus encode/decode at a random bitrate from {6, 12, 24, 48} kbps
  full     — per-sample random choice: 50% none, 25% opus, 25% noise
             (probabilities follow the original v1 recipe)

Every function is seeded through a numpy Generator for reproducibility.
"""

import numpy as np

from src.codec import OPUS_BITRATES, codec_roundtrip


def simulate_resample(audio_array):
    """v1's 'Opus' simulation: plain decimation to 8 kHz and back. NOT real Opus."""
    import librosa

    down = librosa.resample(audio_array, orig_sr=16000, target_sr=8000)
    return librosa.resample(down, orig_sr=8000, target_sr=16000).astype(np.float32)


def add_noise_snr(audio, rng, snr_range=(5.0, 20.0)):
    """Additive white Gaussian noise at a uniformly sampled SNR (dB)."""
    audio = np.asarray(audio, dtype=np.float32)
    snr_db = rng.uniform(*snr_range)
    sig_power = np.mean(audio**2) + 1e-12
    noise_power = sig_power / (10 ** (snr_db / 10.0))
    return audio + rng.normal(0.0, np.sqrt(noise_power), size=len(audio)).astype(np.float32)


def opus_degrade(audio, rng, bitrates=OPUS_BITRATES):
    bitrate = str(bitrates[rng.integers(len(bitrates))])
    return codec_roundtrip(audio, "libopus", bitrate)


AUGMENTATION_MODES = ["none", "noise", "resample", "opus", "full"]


def augment_audio(audio, mode="full", rng=None):
    """Apply the augmentation strategy for `mode` using Generator `rng`."""
    if rng is None:
        rng = np.random.default_rng()
    if mode == "none":
        return np.asarray(audio, dtype=np.float32)
    if mode == "noise":
        return add_noise_snr(audio, rng)
    if mode == "resample":
        return simulate_resample(audio)
    if mode == "opus":
        return opus_degrade(audio, rng)
    if mode == "full":
        u = rng.random()
        if u < 0.5:
            return np.asarray(audio, dtype=np.float32)
        if u < 0.75:
            return opus_degrade(audio, rng)
        return add_noise_snr(audio, rng)
    raise ValueError(f"Unknown augmentation mode: {mode}")
