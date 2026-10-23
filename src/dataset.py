"""Data pipeline v2 — static augmentation pool + tf.data training datasets.

Why a static pool: true-Opus augmentation costs ~100 ms/utterance (ffmpeg
spawn+encode), i.e. ~45 min/epoch serial for the 25,380 training files —
too slow for on-the-fly use inside a Kaggle session. Instead we build, ONCE:

  data/processed/train_waveforms.npy   float16 (N, 65024)   decoded 16 kHz audio
  data/processed/train_mels.npy        float16 (N, 128, 128) log-mels of originals
  data/processed/train_pool.npy        float16 (N, K, 128, 128) augmented mels
  data/processed/train_pool_meta.npz   modes (N, K), params (N, K)

Pool composition (default K=8): 3 true-Opus (random bitrates), 2 naive-resample,
3 Gaussian noise (random SNR 5-20 dB). Per epoch each sample draws randomly
among the eligible entries, so the network still sees a varied mix.

All builders are chunk-resumable (`--start/--end`) so a killed Kaggle session
loses nothing. tf.data is imported lazily — the builders run fine without TF.

Split naming: 'train' | 'dev' | 'eval'.
"""

import os
import sys
import argparse
import threading
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.protocol import CM_PROTOCOLS, CM_AUDIO_DIRS, load_cm_protocol
from src.preprocessing import (
    AUDIO_LEN,
    load_audio,
    waveform_to_logmel,
    normalize_mel,
    load_mel_stats,
)
from src.codec import codec_roundtrip, OPUS_BITRATES
from src.augmentation import simulate_resample, add_noise_snr

PROCESSED = "data/processed"
POOL_K = 8
# Per-file pool layout: 3 opus + 2 resample + 3 noise (see build_aug_pool)
_OPUS_COUNT, _RESAMPLE_COUNT, _NOISE_COUNT = 3, 2, 3


def waveform_cache_path(split):
    return os.path.join(PROCESSED, f"{split}_waveforms.npy")


def mels_path(split):
    return os.path.join(PROCESSED, f"{split}_mels.npy")  # originals (only built for train)


def pool_path(split):
    return os.path.join(PROCESSED, f"{split}_pool.npy")  # train only


def ids_path(split):
    return os.path.join(PROCESSED, f"{split}_ids.json")


def shard_dir(split):
    """Precomputed CLEAN mels for a split, stored in npz shards (for dev/eval)."""
    return os.path.join(PROCESSED, f"{split}_mel_shards")


def get_split_ids(split):
    ids, labels, attacks = load_cm_protocol(CM_PROTOCOLS[split])
    return ids, labels, attacks


# ------------------------------------------------------------------ builders

def _decode_one(args):
    path = args
    return load_audio(path).astype(np.float16)


def build_waveforms(split, workers=8, start=0, end=None, audio_dir=None, out_path=None):
    """Decode all flac files of a split into a float16 memmap (N, 65024).

    `audio_dir` + `out_path` override the standard split directory and cache
    path (used for codec-matrix re-encodes)."""
    ids, _, _ = get_split_ids(split)
    audio_dir = audio_dir or CM_AUDIO_DIRS[split]
    out_path = out_path or waveform_cache_path(split)
    end = end or len(ids)
    os.makedirs(PROCESSED, exist_ok=True)
    arr = np.lib.format.open_memmap(
        out_path, mode="r+" if os.path.exists(out_path) else "w+",
        dtype=np.float16, shape=(len(ids), AUDIO_LEN),
    )
    jobs = [(i, os.path.join(audio_dir, f"{fid}.flac")) for i, fid in enumerate(ids) if start <= i < end]
    with Pool(workers) as p:
        for n, wav in enumerate(p.imap(_decode_one, [j[1] for j in jobs], chunksize=16)):
            arr[jobs[n][0]] = wav
            if (n + 1) % 1000 == 0:
                print(f"  waveforms {n + 1}/{len(jobs)}", flush=True)
    arr.flush()
    print(f"done: {out_path} rows [{start}, {end})")


def _mel_from_cache(args):
    i, cache_path = args
    arr = np.load(cache_path, mmap_mode="r")
    return i, waveform_to_logmel(arr[i].astype(np.float32)).astype(np.float16)


def build_original_mels(split, workers=8, start=0, end=None, cache_path=None):
    """Log-mels of the ORIGINAL (clean) waveforms, float16 (N, 128, 128)."""
    cache_path = cache_path or waveform_cache_path(split)
    ids, _, _ = get_split_ids(split)
    end = end or len(ids)
    arr = np.lib.format.open_memmap(
        mels_path(split), mode="r+" if os.path.exists(mels_path(split)) else "w+",
        dtype=np.float16, shape=(len(ids), 128, 128),
    )
    jobs = [(i, cache_path) for i in range(start, end)]
    with Pool(workers) as p:
        for n, (i, mel) in enumerate(p.imap(_mel_from_cache, jobs, chunksize=8)):
            arr[i] = mel
            if (n + 1) % 2000 == 0:
                print(f"  mels {n + 1}/{len(jobs)}", flush=True)
    arr.flush()
    print(f"done: {mels_path(split)} rows [{start}, {end})")


def _augment_row(args):
    """Worker: build the K augmented mels for one utterance."""
    i, cache_path, k, seed = args
    rng = np.random.default_rng(seed + i)
    wav = np.load(cache_path, mmap_mode="r")[i].astype(np.float32)
    mels = np.zeros((k, 128, 128), dtype=np.float16)
    modes = np.empty(k, dtype="U8")
    params = np.empty(k, dtype="U8")
    j = 0
    for _ in range(_OPUS_COUNT):
        br = str(OPUS_BITRATES[rng.integers(len(OPUS_BITRATES))])
        mels[j] = waveform_to_logmel(codec_roundtrip(wav, "libopus", br)).astype(np.float16)
        modes[j], params[j] = "opus", br
        j += 1
    for _ in range(_RESAMPLE_COUNT):
        mels[j] = waveform_to_logmel(simulate_resample(wav)).astype(np.float16)
        modes[j], params[j] = "resample", "8k"
        j += 1
    for _ in range(_NOISE_COUNT):
        snr = rng.uniform(5.0, 20.0)
        noisy = add_noise_snr(wav, rng, snr_range=(snr, snr))
        mels[j] = waveform_to_logmel(noisy).astype(np.float16)
        modes[j], params[j] = "noise", f"{snr:.2f}"
        j += 1
    return i, mels, modes, params


def build_aug_pool(split, k=POOL_K, workers=8, start=0, end=None, seed=1234):
    """The static augmentation pool, float16 (N, K, 128, 128). Chunk-resumable."""
    cache_path = waveform_cache_path(split)
    ids, _, _ = get_split_ids(split)
    end = end or len(ids)
    assert k == _OPUS_COUNT + _RESAMPLE_COUNT + _NOISE_COUNT, "k must match pool layout"
    arr = np.lib.format.open_memmap(
        pool_path(split), mode="r+" if os.path.exists(pool_path(split)) else "w+",
        dtype=np.float16, shape=(len(ids), k, 128, 128),
    )
    meta_file = os.path.join(PROCESSED, f"{split}_pool_meta.npz")
    modes = np.empty((len(ids), k), dtype="U8")
    params = np.empty((len(ids), k), dtype="U8")
    if os.path.exists(meta_file) and start != 0:
        prev = np.load(meta_file)
        modes[:], params[:] = prev["modes"], prev["params"]
    jobs = [(i, cache_path, k, seed) for i in range(start, end)]
    with Pool(workers) as p:
        for n, (i, mels, m, q) in enumerate(p.imap(_augment_row, jobs, chunksize=4)):
            arr[i] = mels
            modes[i], params[i] = m, q
            if (n + 1) % 500 == 0:
                print(f"  pool {n + 1}/{len(jobs)}", flush=True)
    arr.flush()
    np.savez_compressed(meta_file, modes=modes, params=params)
    print(f"done: {pool_path(split)} rows [{start}, {end})")


# ------------------------------------------------------- clean-mel npz shards

def _mel_direct(path):
    return waveform_to_logmel(load_audio(path)).astype(np.float16)


def build_clean_mel_shards(split, shard_size=5000, workers=8, start=0, end=None,
                           audio_dir=None, out_dir=None, cache_path=None):
    """Precompute clean mels for a split (dev/eval/codec-matrix) into npz shards.

    If `cache_path` is given, mels are computed from that waveform cache;
    otherwise flacs are decoded directly (used for dev/eval to avoid keeping
    a 9 GB waveform cache on disk — /kaggle/working has a ~20 GB cap)."""
    direct_dir = audio_dir or CM_AUDIO_DIRS[split]
    ids, labels, attacks = get_split_ids(split)
    end = end or len(ids)
    out_dir = out_dir or shard_dir(split)
    os.makedirs(out_dir, exist_ok=True)
    jobs = [(i, cache_path) for i in range(start, end)] if cache_path else \
           [os.path.join(direct_dir, f"{fid}.flac") for fid in ids[start:end]]
    worker = _mel_from_cache if cache_path else _mel_direct
    shard_id = start // shard_size
    for chunk_start in range(start, end, shard_size):
        chunk_end = min(chunk_start + shard_size, end)
        shard_file = os.path.join(out_dir, f"shard_{chunk_start:06d}.npz")
        if os.path.exists(shard_file):
            continue
        with Pool(workers) as p:
            if cache_path:
                results = p.map(worker, [(i, cache_path) for i in range(chunk_start, chunk_end)], chunksize=8)
                mels = np.zeros((chunk_end - chunk_start, 128, 128), dtype=np.float16)
                for i, mel in results:
                    mels[i - chunk_start] = mel
            else:
                mels = np.stack(p.map(worker, jobs[chunk_start - start:chunk_end - start], chunksize=8))
        np.savez_compressed(
            shard_file,
            mels=mels,
            ids=np.array(ids[chunk_start:chunk_end]),
            labels=labels[chunk_start:chunk_end],
            attacks=np.array(attacks[chunk_start:chunk_end]),
        )
        shard_id += 1
        print(f"  shard {shard_file} done", flush=True)
    print(f"done: shards in {out_dir}")


def load_clean_mels(split):
    """Load all precomputed clean-mel shards for a split.

    Returns (mels float16 (N,128,128), ids, labels, attacks)."""
    out_dir = shard_dir(split)
    files = sorted(f for f in os.listdir(out_dir) if f.endswith(".npz"))
    if not files:
        raise FileNotFoundError(f"no shards in {out_dir} — run build_clean_mel_shards first")
    mels, ids, labels, attacks = [], [], [], []
    for f in files:
        d = np.load(os.path.join(out_dir, f), allow_pickle=False)
        mels.append(d["mels"])
        ids.extend(d["ids"].tolist())
        labels.append(d["labels"])
        attacks.extend(d["attacks"].tolist())
    return (
        np.concatenate(mels),
        ids,
        np.concatenate(labels),
        attacks,
    )


# ------------------------------------------------------------- tf.data layer

_fetch_lock = threading.Lock()


def _make_fetch(mode, split="train"):
    """Build the per-sample fetch used inside tf.numpy_function.

    Returns fn(idx) -> flat float32 (128*128,) normalized mel.
    Dynamic randomness (which pool entry per epoch) via a lock-guarded RNG.
    """
    orig = np.load(mels_path(split), mmap_mode="r")
    pool = np.load(pool_path(split), mmap_mode="r") if mode != "none" else None
    pool_modes = None
    if pool is not None:
        meta = np.load(os.path.join(PROCESSED, f"{split}_pool_meta.npz"))
        pool_modes = meta["modes"]
    stats = load_mel_stats()
    mean, std = stats if stats is not None else (0.0, 1.0)
    rng = np.random.default_rng(0)
    warned = set()

    def _pick(i, want):
        cand = np.where(pool_modes[i] == want)[0]
        if len(cand) == 0:
            # pool gap (e.g. partial build) — fall back to the original mel
            if want not in warned:
                print(f"[dataset] WARNING: no '{want}' pool entries for row {i}; "
                      "falling back to original mel (pool build incomplete?)", flush=True)
                warned.add(want)
            return np.asarray(orig[i])
        return np.asarray(pool[i, cand[rng.integers(len(cand))]])

    def fetch(idx):
        i = int(idx)
        with _fetch_lock:
            if mode == "none":
                m = np.asarray(orig[i])
            elif mode in ("noise", "resample", "opus"):
                m = _pick(i, mode)
            elif mode == "full":
                u = rng.random()
                m = np.asarray(orig[i]) if u < 0.5 else _pick(i, "opus" if u < 0.75 else "noise")
            else:
                raise ValueError(mode)
        m = m.astype(np.float32)
        return ((m - mean) / std).ravel()

    return fetch


def make_train_dataset(mode, split="train", batch_size=32, seed=42):
    """tf.data training dataset: infinite, shuffled, batched, preloaded from pool."""
    import tensorflow as tf

    ids, labels, _ = get_split_ids(split)
    n = len(ids)
    fetch = _make_fetch(mode, split)

    def _fn(idx, y):
        [mel] = tf.numpy_function(fetch, [idx], [tf.float32])
        mel = tf.reshape(mel, (128, 128, 1))
        return mel, tf.cast(y, tf.float32)

    ds = tf.data.Dataset.from_tensor_slices((np.arange(n, dtype=np.int64), labels.astype(np.float32)))
    ds = ds.shuffle(n, seed=seed, reshuffle_each_iteration=True).repeat()
    ds = ds.map(_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, n


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build v2 data pipeline artifacts")
    parser.add_argument("cmd", choices=["build-waveforms", "build-mels", "build-pool", "build-shards"])
    parser.add_argument("--split", default="train", choices=["train", "dev", "eval"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    args = parser.parse_args()
    if args.cmd == "build-waveforms":
        build_waveforms(args.split, args.workers, args.start, args.end)
    elif args.cmd == "build-mels":
        build_original_mels(args.split, args.workers, args.start, args.end)
    elif args.cmd == "build-pool":
        build_aug_pool(args.split, workers=args.workers, start=args.start, end=args.end)
    elif args.cmd == "build-shards":
        build_clean_mel_shards(args.split, workers=args.workers, start=args.start, end=args.end)
