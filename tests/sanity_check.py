"""Sanity tests for the v2 pipeline (no TensorFlow required).

Run: python tests/sanity_check.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from src.metrics import compute_eer, compute_min_tdcf, compute_asv_operating_point, full_metrics
from src.preprocessing import (
    AUDIO_LEN,
    load_audio,
    waveform_to_logmel,
    normalize_mel,
    audio_to_mel,
)
from src.codec import OPUS_BITRATES, codec_roundtrip
from src.augmentation import AUGMENTATION_MODES, augment_audio

PASS = "  PASS"
FAIL = "! FAIL"
failures = []


def check(name, cond, extra=""):
    print(f"{PASS if cond else FAIL}  {name} {extra}")
    if not cond:
        failures.append(name)


# ---------------------------------------------------------------- metrics
print("== metrics ==")
rng = np.random.default_rng(0)
pos = rng.normal(1.0, 1.0, 500)   # bonafide scores (higher = bonafide)
neg = rng.normal(0.0, 1.0, 500)   # spoof scores

eer, thr = compute_eer(pos, neg)
check("EER ~ 0.24 for 1-sigma separation", 0.15 < eer < 0.35, f"(eer={eer:.3f})")

eer_same, _ = compute_eer(pos, pos.copy())
check("EER ~ 0.5 for identical distributions", 0.4 < eer_same < 0.6, f"(eer={eer_same:.3f})")

eer_perfect, _ = compute_eer(np.array([0.1, 0.2, 0.3]), np.array([-0.1, -0.2, -0.3]))
check("EER ~ 0 for perfectly separable", eer_perfect < 1e-3, f"(eer={eer_perfect:.5f})")

asv = {
    "tar_asv": pos,
    "non_asv": rng.normal(1.0, 1.0, 500),  # nontarget scores: also bonafide-ish speakers
    "spoof_asv": neg,
}
ap = compute_asv_operating_point(asv["tar_asv"], asv["non_asv"], asv["spoof_asv"])
check("ASV op-point in (0,1)", all(0 <= v <= 1 for k, v in ap.items() if k != "asv_threshold"))

labels = np.concatenate([np.ones(500), np.zeros(500)])
scores = np.concatenate([pos, neg])
min_tdcf, info = compute_min_tdcf(scores, labels, asv=asv)
check("min t-DCF in [0, ~1] for decent CM", 0.0 <= min_tdcf <= 1.2, f"(t-DCF={min_tdcf:.3f})")

# trivial CM: all bonafide (accept everything) -> t-DCF_norm should equal 1.0
trivial_scores = np.ones_like(scores)
t_trivial, _ = compute_min_tdcf(trivial_scores, labels, asv_params=ap)
check("trivial always-accept CM -> t-DCF_norm = 1.0", abs(t_trivial - 1.0) < 1e-9, f"(t={t_trivial:.6f})")

perfect_scores = np.concatenate([np.ones(500), -np.ones(500)])
t_perf, _ = compute_min_tdcf(perfect_scores, labels, asv_params=ap)
check("perfect CM -> t-DCF_norm = 0", t_perf < 1e-9, f"(t={t_perf:.2e})")

attacks = ["-"] * 500 + ["A07"] * 250 + ["A08"] * 250
pa = full_metrics(scores, labels, attacks=attacks, asv=asv)
check("per-attack EER present for A07/A08", set(pa["per_attack"]) == {"A07", "A08"})

boot = full_metrics(scores, labels, asv=asv, bootstrap=True)["eer_bootstrap"]
check("bootstrap CI brackets EER", boot["ci_lo"] <= eer <= boot["ci_hi"], f"({boot})")

# ---------------------------------------------------------- preprocessing
print("== preprocessing ==")
test_file = "data/raw/LA/LA/ASVspoof2019_LA_train/flac/LA_T_1000648.flac"
w = load_audio(test_file)
check("load_audio length = 65024", len(w) == AUDIO_LEN, f"({len(w)})")

mel = waveform_to_logmel(w)
check("mel shape (128, 128)", mel.shape == (128, 128), f"({mel.shape})")

norm = normalize_mel(mel)
check("normalized mel is finite", np.all(np.isfinite(norm)))

mel2 = audio_to_mel(w)
check("audio_to_mel shape ok", mel2.shape == (128, 128))

# ----------------------------------------------------------------- codecs
print("== codecs ==")
sig = (0.5 * np.sin(2 * np.pi * 220 * np.linspace(0, 1.0, 16000)) + 0.05 * rng.normal(size=16000)).astype(np.float32)
for codec, br in [("libopus", "24000"), ("libmp3lame", "64k"), ("aac", "64k"), ("libopencore_amrnb", "12.2k")]:
    try:
        out = codec_roundtrip(sig, codec, br)
        corr = np.corrcoef(sig, out)[0, 1]
        check(f"codec roundtrip {codec}@{br} preserved", out.shape == sig.shape and corr > 0.9, f"(corr={corr:.3f})")
    except Exception as e:
        check(f"codec roundtrip {codec}@{br}", False, f"({e})")

# ------------------------------------------------------------ augmentation
print("== augmentation ==")
for mode in AUGMENTATION_MODES:
    out = augment_audio(sig, mode=mode, rng=np.random.default_rng(1))
    check(f"augment mode={mode} returns finite same-length", out.shape == sig.shape and np.all(np.isfinite(out)))

out_opus = augment_audio(sig, mode="opus", rng=np.random.default_rng(2))
corr = np.corrcoef(sig, out_opus)[0, 1]
check("opus degradation stays correlated", 0.7 < corr <= 1.01, f"(corr={corr:.3f})")

out_res = augment_audio(sig, mode="resample", rng=np.random.default_rng(2))
corr_r = np.corrcoef(sig, out_res)[0, 1]
check("resample degradation stays correlated", 0.7 < corr_r, f"(corr={corr_r:.3f})")

# opus timing (drives the per-epoch estimate on Kaggle)
t0 = time.time()
for _ in range(10):
    augment_audio(np.tile(sig, 5)[:65024].copy(), mode="opus", rng=rng)
dt = (time.time() - t0) / 10
print(f"  INFO: opus aug per 4s utterance: {dt * 1000:.0f} ms -> 25380 files/epoch ~ {25380 * dt / 60:.1f} min serial")

print()
if failures:
    print(f"{len(failures)} FAILURES: {failures}")
    sys.exit(1)
print("ALL SANITY CHECKS PASSED")
