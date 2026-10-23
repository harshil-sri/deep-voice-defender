"""Kaggle-oriented trainer for Deep-Voice Defender v2.

Key properties:
  - Trains from the static augmentation pool (src/dataset.py) — no ffmpeg/librosa
    in the training loop, so epochs are fast and session-kill safe.
  - Model selection on DEV EER (early stopping) instead of training loss —
    the ASVspoof-standard protocol (train / dev / eval are strictly separate).
  - Fully resumable: per-epoch checkpoint + CSV log. If a Kaggle 12 h session
    dies, re-running the same command continues from the last completed epoch.

Usage:
  python src/train_kaggle.py --aug_mode full --seed 42
  python src/train_kaggle.py --aug_mode none --seed 123 --time_budget_h 8
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402

from src.model import build_model  # noqa: E402
from src.dataset import make_train_dataset  # noqa: E402
from src.metrics import compute_eer  # noqa: E402

EPOCHS_DEFAULT = 60
PATIENCE_DEFAULT = 10


def evaluate_dev_eer(model, dev_mels, dev_labels, batch_size=128, stats=None):
    """Pooled EER on precomputed clean dev mels (float16, raw log-mel domain)."""
    from src.preprocessing import load_mel_stats

    if stats is None:
        stats = load_mel_stats() or (0.0, 1.0)
    scores = []
    mean, std = stats if stats is not None else (0.0, 1.0)
    for i in range(0, len(dev_labels), batch_size):
        batch = dev_mels[i : i + batch_size].astype(np.float32)
        batch = (batch - mean) / std
        batch = np.expand_dims(batch, -1)
        scores.append(model.predict(batch, verbose=0).ravel())
    scores = np.concatenate(scores)
    eer, _ = compute_eer(scores[dev_labels == 1], scores[dev_labels == 0])
    return float(eer)


def load_dev_data():
    from src.dataset import load_clean_mels

    mels, _, labels, _ = load_clean_mels("dev")
    return mels, np.asarray(labels)


def train_run(aug_mode, seed, epochs=EPOCHS_DEFAULT, batch_size=32, output_dir=None,
              patience=PATIENCE_DEFAULT, time_budget_h=None, class_weights=False,
              learning_rate=2e-4, run_name=None):
    run_name = run_name or f"dvd_{aug_mode}_seed{seed}"
    output_dir = output_dir or os.environ.get("DVD_OUTPUT_DIR", "models")
    run_dir = os.path.join(output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    ckpt_path = os.path.join(run_dir, "ckpt_latest.keras")
    best_path = os.path.join(run_dir, "best.keras")
    csv_path = os.path.join(run_dir, "log.csv")
    state_path = os.path.join(run_dir, "state.json")

    np.random.seed(seed)
    tf.random.set_seed(seed)

    print(f"[{run_name}] building tf.data pipeline (mode={aug_mode})...")
    train_ds, n_train = make_train_dataset(aug_mode, split="train", batch_size=batch_size, seed=seed)
    steps_per_epoch = n_train // batch_size

    print(f"[{run_name}] loading dev mels for model selection...")
    dev_mels, dev_labels = load_dev_data()
    print(f"[{run_name}] dev: {len(dev_labels)} utterances")

    model = build_model((128, 128, 1))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )

    # ---- resume state
    start_epoch = 0
    best_eer = float("inf")
    best_epoch = -1
    history = []
    if os.path.exists(state_path):
        with open(state_path) as f:
            state = json.load(f)
        start_epoch = state["last_epoch"] + 1
        best_eer = state["best_eer"]
        best_epoch = state["best_epoch"]
        history = state.get("history", [])
        print(f"[{run_name}] resuming at epoch {start_epoch} (best dev EER {best_eer:.4f} @ {best_epoch})")
    elif os.path.exists(csv_path):
        print(f"[{run_name}] WARNING: log.csv exists without state.json — restarting clean")
        os.remove(csv_path)

    if start_epoch > 0:
        model = tf.keras.models.load_model(ckpt_path, compile=True)

    cw = {0: 1.0, 1: 9.0} if class_weights else None
    t_start = time.time()
    deadline = t_start + time_budget_h * 3600 if time_budget_h else None

    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        hist = model.fit(
            train_ds,
            steps_per_epoch=steps_per_epoch,
            epochs=epoch + 1,
            initial_epoch=epoch,
            class_weight=cw,
            verbose=2,
        )
        train_loss = float(hist.history["loss"][0])

        dev_eer = evaluate_dev_eer(model, dev_mels, dev_labels, stats=None)  # stats loaded inside loader? no — see note
        improved = dev_eer < best_eer
        if improved:
            best_eer, best_epoch = dev_eer, epoch
            model.save(best_path)

        model.save(ckpt_path)  # per-epoch checkpoint (session-kill safe)
        history.append({
            "epoch": epoch,
            "loss": train_loss,
            "dev_eer": dev_eer,
            "lr": float(model.optimizer.learning_rate.numpy()),
            "sec": time.time() - t0,
        })
        with open(state_path, "w") as f:
            json.dump({
                "run_name": run_name, "aug_mode": aug_mode, "seed": seed,
                "last_epoch": epoch, "best_eer": best_eer, "best_epoch": best_epoch,
                "history": history,
                "updated": datetime.now().isoformat(),
            }, f, indent=2)
        # csv mirror for easy plotting
        if epoch == start_epoch and not os.path.exists(csv_path):
            with open(csv_path, "w") as f:
                f.write("epoch,loss,dev_eer,sec\n")
        with open(csv_path, "a") as f:
            f.write(f"{epoch},{train_loss:.5f},{dev_eer:.5f},{history[-1]['sec']:.1f}\n")

        marker = " *" if improved else ""
        print(f"[{run_name}] epoch {epoch}: loss={train_loss:.4f} dev_EER={dev_eer * 100:.2f}%{marker}"
              f" (best {best_eer * 100:.2f}% @ {best_epoch}) {history[-1]['sec']:.0f}s")

        if epoch - best_epoch >= patience:
            print(f"[{run_name}] early stopping (no dev improvement for {patience} epochs)")
            break
        if deadline and time.time() > deadline:
            print(f"[{run_name}] time budget reached — stopping cleanly (resumable)")
            break

    model.save(os.path.join(run_dir, "final.keras"))
    print(f"[{run_name}] DONE. best dev EER={best_eer * 100:.2f}% @ epoch {best_epoch}")
    return best_eer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--aug_mode", required=True,
                        choices=["none", "noise", "resample", "opus", "full"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=PATIENCE_DEFAULT)
    parser.add_argument("--time_budget_h", type=float, default=None,
                        help="stop cleanly after this many hours (for Kaggle session limits)")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--class_weights", action="store_true")
    parser.add_argument("--run_name", default=None)
    args = parser.parse_args()
    train_run(args.aug_mode, args.seed, epochs=args.epochs, batch_size=args.batch_size,
              patience=args.patience, time_budget_h=args.time_budget_h,
              output_dir=args.output_dir, class_weights=args.class_weights,
              run_name=args.run_name)
