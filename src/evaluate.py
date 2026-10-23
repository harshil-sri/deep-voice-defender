"""Evaluation v2 — model × condition scoring with OFFICIAL ASVspoof 2019 metrics.

Outputs per (model, split, condition):
  results/scores/{run}__{split}__{condition}.csv   raw per-utterance scores
      (columns: utt_id,label,attack,score) — the artifact ALL later analysis
      (per-attack tables, bootstrap CIs, codec-robustness matrices) is built from
  results/eval/{run}__{split}__{condition}.json    metric summary

Inputs are PRECOMPUTED clean/codec-matrix mels (from src/dataset.py
build-shards / scripts/precompute_mels.py) — scoring 71k utterances is then
a couple of minutes on GPU instead of ~20.

Examples:
  python src/evaluate.py --model models/dvd_full_seed42/best.keras --split eval --condition clean
  python src/evaluate.py --model ... --split eval --condition opus_12k \
      --mels_dir data/processed/codec_opus_12k_eval_shards
"""

import os
import sys
import csv
import json
import argparse
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402

from src.protocol import load_cm_protocol, load_asv_trials, CM_PROTOCOLS, ASV_PROTOCOLS, ASV_SCORES  # noqa: E402
from src.metrics import full_metrics  # noqa: E402
from src.preprocessing import load_mel_stats  # noqa: E402


def load_mels_from_dir(mels_dir):
    """Load precomputed mel shards (same format as src/dataset.load_clean_mels)."""
    files = sorted(f for f in os.listdir(mels_dir) if f.endswith(".npz"))
    if not files:
        raise FileNotFoundError(f"no .npz shards in {mels_dir}")
    mels, ids, labels, attacks = [], [], [], []
    for f in files:
        d = np.load(os.path.join(mels_dir, f))
        mels.append(d["mels"])
        ids.extend(d["ids"].tolist())
        labels.append(d["labels"])
        attacks.extend(d["attacks"].tolist())
    return np.concatenate(mels), ids, np.concatenate(labels), attacks


def predict_scores(model, mels, stats, batch_size=128):
    """mels: float16 (N,128,128) raw log-mel -> model scores (N,), high = bonafide."""
    mean, std = stats
    scores = []
    for i in range(0, len(mels), batch_size):
        batch = mels[i : i + batch_size].astype(np.float32)
        batch = ((batch - mean) / std)[..., None]
        scores.append(model.predict(batch, verbose=0).ravel())
    return np.concatenate(scores)


def evaluate(model_path, split, condition, mels_dir=None, protocol=None,
             asv_score_file=None, asv_protocol=None, out_root="results",
             bootstrap=False, batch_size=128):
    run = os.path.basename(model_path).replace(".keras", "")
    tag = f"{run}__{split}__{condition}"

    if mels_dir is None:
        from src.dataset import shard_dir
        mels_dir = shard_dir(split)
    mels, ids, labels, attacks = load_mels_from_dir(mels_dir)

    protocol = protocol or CM_PROTOCOLS[split]
    proto_ids, proto_labels, proto_attacks = load_cm_protocol(protocol)
    # shards are built in protocol order; verify rather than re-index (O(N^2) trap)
    if len(ids) != len(proto_ids) or ids != proto_ids:
        if set(ids) != set(proto_ids):
            raise RuntimeError("shard utterances do not match the protocol")

    model = tf.keras.models.load_model(model_path, compile=False)
    stats = load_mel_stats() or (0.0, 1.0)
    scores = predict_scores(model, mels, stats, batch_size=batch_size)

    # write raw scores CSV first (never lose raw scores, even if metrics fail)
    scores_dir = os.path.join(out_root, "scores")
    os.makedirs(scores_dir, exist_ok=True)
    csv_path = os.path.join(scores_dir, f"{tag}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["utt_id", "label", "attack", "score"])
        for u, l, a, s in zip(ids, labels, attacks, scores):
            w.writerow([u, int(l), a, f"{s:.6f}"])
    print(f"scores -> {csv_path}")

    # official metrics (needs the ASV scores of the same split)
    asv = None
    if split in ASV_SCORES and os.path.exists(ASV_SCORES[split]):
        asv = load_asv_trials(ASV_SCORES[split], ASV_PROTOCOLS[split])
    metrics = full_metrics(scores, labels, attacks=attacks, asv=asv, bootstrap=bootstrap)

    out = {
        "run": run,
        "split": split,
        "condition": condition,
        "model_path": model_path,
        "mels_dir": mels_dir,
        "protocol": protocol,
        "num_samples": int(len(ids)),
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics,
        "scores_csv": csv_path,
    }
    eval_dir = os.path.join(out_root, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    json_path = os.path.join(eval_dir, f"{tag}.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"metrics -> {json_path}")
    print(f"  EER      {metrics['eer'] * 100:.2f}%")
    print(f"  min t-DCF {metrics['min_tDCF']:.4f}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", default="eval", choices=["train", "dev", "eval"])
    parser.add_argument("--condition", default="clean",
                        help="e.g. clean, opus_12k, mp3_32k, asvspoof2021")
    parser.add_argument("--mels_dir", default=None,
                        help="shard dir with precomputed mels (default: built-in split shards)")
    parser.add_argument("--protocol", default=None, help="override CM protocol file")
    parser.add_argument("--asv_scores", default=None)
    parser.add_argument("--asv_protocol", default=None)
    parser.add_argument("--out_root", default="results")
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()
    evaluate(args.model, args.split, args.condition, mels_dir=args.mels_dir,
             protocol=args.protocol, asv_score_file=args.asv_scores,
             asv_protocol=args.asv_protocol, out_root=args.out_root,
             bootstrap=args.bootstrap, batch_size=args.batch_size)
