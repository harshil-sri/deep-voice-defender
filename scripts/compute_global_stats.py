"""Compute global mel statistics (mean/std of log-mel) from the ASVspoof 2019 LA train set.

Uses a random sample of train utterances (default 5000) — enough for a scalar
normalization. Output: data/processed/mel_stats.npz (consumed by every v2
module; the two numbers ARE part of the model spec now).

Run AFTER build-waveforms if the waveform cache exists (fast path), otherwise
decodes the sampled flacs directly.
"""

import os
import sys
import argparse
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.protocol import CM_PROTOCOLS, CM_AUDIO_DIRS, load_cm_protocol
from src.preprocessing import waveform_to_logmel, load_audio, load_mel_stats


def _mel_one(path):
    return waveform_to_logmel(load_audio(path))


def main(sample_n=5000, workers=8, seed=0):
    out = os.path.join("data", "processed", "mel_stats.npz")
    os.makedirs("data/processed", exist_ok=True)

    if os.path.exists("data/processed/train_mels.npy"):
        mels = np.load("data/processed/train_mels.npy", mmap_mode="r")
        rng = np.random.default_rng(seed)
        idx = rng.choice(mels.shape[0], size=min(sample_n, mels.shape[0]), replace=False)
        sample = np.stack([np.asarray(mels[i]) for i in idx])
        print(f"sampled {len(idx)} from train_mels.npy")
    else:
        ids, _, _ = load_cm_protocol(CM_PROTOCOLS["train"])
        rng = np.random.default_rng(seed)
        chosen = rng.choice(ids, size=min(sample_n, len(ids)), replace=False)
        paths = [os.path.join(CM_AUDIO_DIRS["train"], f"{u}.flac") for u in chosen]
        with Pool(workers) as p:
            sample = np.stack(p.map(_mel_one, paths, chunksize=8))
        print(f"decoded+mel'd {len(paths)} train files")

    mean = float(np.mean(sample))
    std = float(np.std(sample))
    np.savez(out, mean=mean, std=std, n=int(sample.shape[0]))
    print(f"mel stats: mean={mean:.4f} std={std:.4f} -> {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_n", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    main(args.sample_n, args.workers)
