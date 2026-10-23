"""Preprocessing v2 — fixed-length waveforms -> (128, 128) log-mel -> global-stat normalization.

v2 changes vs the original pipeline (all documented in paper/PLAN.md):
  - Waveforms are fixed at 65024 samples so that, with hop=512, the mel
    spectrogram is EXACTLY (128, 128) — the model input is now truly that shape.
  - Normalization switched from per-utterance min-max to GLOBAL statistics
    (mean/std of log-mel over the ASVspoof 2019 LA train set), stored in
    data/processed/mel_stats.npz by scripts/compute_global_stats.py.
    Per-utterance min-max destroys absolute level information and lets
    near-silent frames define the scale.
  - `audio_to_mel` returns the globally-normalized mel by default; if the
    stats file is missing it falls back to the legacy min-max behavior so the
    Streamlit app keeps running.
"""

import os

import librosa
import numpy as np

SAMPLE_RATE = 16000
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512
# (N_MELS?) -> with hop 512, 128 frames need 65024 samples: 1 + 65024//512 = 128
AUDIO_LEN = (N_MELS - 1) * HOP_LENGTH  # 65024
MEL_STATS_PATH = os.path.join("data", "processed", "mel_stats.npz")


def load_audio(file_path, target_len=AUDIO_LEN, sr=SAMPLE_RATE):
    """Load audio as float32 mono at 16 kHz, pad with zeros / trim to target_len."""
    audio, _ = librosa.load(file_path, sr=sr, mono=True)
    audio = audio.astype(np.float32)
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)), mode="constant")
    elif len(audio) > target_len:
        audio = audio[:target_len]
    return audio


def pad_or_trim(waveform, target_len=AUDIO_LEN):
    waveform = np.asarray(waveform, dtype=np.float32)
    if len(waveform) < target_len:
        return np.pad(waveform, (0, target_len - len(waveform)), mode="constant")
    return waveform[:target_len]


def waveform_to_logmel(waveform):
    """Raw (unnormalized) log-mel spectrogram, shape (128, 128)."""
    mel = librosa.feature.melspectrogram(
        y=waveform, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)


def load_mel_stats(stats_path=MEL_STATS_PATH):
    if not os.path.exists(stats_path):
        return None
    d = np.load(stats_path)
    return float(d["mean"]), float(d["std"])


def normalize_mel(logmel, stats=None):
    """Global-stat normalization: (x - mean) / std. Falls back to legacy min-max."""
    if stats is None:
        stats = load_mel_stats()
    if stats is not None:
        mean, std = stats
        return ((logmel - mean) / std).astype(np.float32)
    # Legacy per-utterance min-max (used only when global stats are unavailable)
    return ((logmel - np.min(logmel)) / (np.max(logmel) - np.min(logmel) + 1e-8)).astype(np.float32)


def audio_to_mel(waveform, stats=None):
    """Waveform -> normalized (128, 128) mel. Global stats by default."""
    return normalize_mel(waveform_to_logmel(pad_or_trim(waveform)), stats=stats)


def audio_to_mel_minmax(waveform):
    """Legacy v1 preprocessing (per-utterance min-max), kept for app.py compatibility."""
    logmel = waveform_to_logmel(pad_or_trim(waveform))
    return ((logmel - np.min(logmel)) / (np.max(logmel) - np.min(logmel) + 1e-8)).astype(np.float32)
