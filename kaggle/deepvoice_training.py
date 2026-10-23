"""
DeepVoice Defender v2 — Kaggle kernel entry point.

Attach as inputs (edit kernel-metadata.json):
  - asvspoof2019-la            (private dataset uploaded from data/raw/LA/LA)
  - dvd-processed              (created once from the output of the `prep` run —
                                waveform caches, augmentation pool, mel shards)
  - deepvoice-defender-code    (optional: this repo as a dataset; else the
                                script clones it from GitHub)

Stages (run via the last cell / __main__):
  prep   — build waveform caches, original mels, augmentation pool, global mel
           stats, clean mel shards for dev+eval.  CPU-only, ~3-4 h, resumable.
  train  — train one run:            train --mode full --seed 42
  grid   — train everything still missing, highest-priority seeds first,
           stops cleanly at --time_budget_h.
  eval   — score every finished model on dev+eval (clean) and the codec matrix.

Artifacts land in /kaggle/working/models and /kaggle/working/results.
"""

import os
import sys
import subprocess
import argparse

REPO = os.environ.get("DVD_REPO", "/kaggle/working/repo")
if not os.path.isdir(REPO):
    for cand in ["/kaggle/input/deepvoice-defender-code", "/kaggle/input/deep-voice-defender"]:
        if os.path.isdir(cand):
            subprocess.run(["cp", "-r", cand, REPO], check=False)
            break
    else:
        subprocess.run(["git", "clone", "https://github.com/harshil-sri/deep-voice-defender.git", REPO], check=True)

os.chdir(REPO)
sys.path.insert(0, REPO)
os.environ["DVD_OUTPUT_DIR"] = "/kaggle/working/models"
os.makedirs("/kaggle/working/models", exist_ok=True)
os.makedirs("/kaggle/working/results", exist_ok=True)

# Wire the ASVspoof data. Public mirror anishsarkar22/asvpoof-2019-dataset-la
# keeps the official zip layout, so the real root is one level down (…/LA).
# A future private upload (mrigankrana/asvspoof2019-la) has the corpus at the root.
KAGGLE_DATA = None
for root, sub in [
    ("/kaggle/input/asvspoof2019-la", ""),                     # own upload (corpus at root)
    ("/kaggle/input/asvpoof-2019-dataset-la", "/LA"),          # public mirror
    ("/kaggle/input/asvspoof2019-la-1", "/LA"),                # alternate mirror naming
]:
    cand = os.path.join(root, sub) if sub else root
    if os.path.isdir(os.path.join(cand, "ASVspoof2019_LA_cm_protocols")):
        KAGGLE_DATA = cand
        break
if KAGGLE_DATA:
    link = os.path.join(REPO, "data", "raw", "LA", "LA")
    os.makedirs(os.path.dirname(link), exist_ok=True)
    if not os.path.exists(link):
        os.symlink(KAGGLE_DATA, link)
    print(f"data -> {KAGGLE_DATA}")

# Wire precomputed processed artifacts (from the prep run's promoted output).
# The dataset may nest them at any depth (repo/data/processed if created from
# the kernel Output tab, processed/ if created from a restructured download).
import glob as _glob

GRID_PRIORITY = [(m, s) for s in (42, 123, 456) for m in ("full", "none", "opus", "resample", "noise")]

EPOCHS_DEFAULT = 60
PATIENCE_DEFAULT = 10

for pat in ("/kaggle/input/*/processed", "/kaggle/input/*/data/processed",
            "/kaggle/input/*/repo/data/processed"):
    hit = sorted(_glob.glob(pat))
    if hit:
        link = os.path.join(REPO, "data", "processed")
        if not os.path.exists(link):
            os.symlink(hit[0], link)
        print(f"processed -> {hit[0]}")
        break

# Seed models/results from previous sessions (dvd-grid-state dataset) so the
# grid and eval stages see everything that finished before this session.
for src in sorted(_glob.glob("/kaggle/input/*/models")) + sorted(_glob.glob("/kaggle/input/*/results")):
    dst = os.path.join("/kaggle/working", os.path.basename(src))
    subprocess.run(["cp", "-rn", src + "/.", dst], check=False)
    print(f"seeded {dst} <- {src}")


def run_finished(run_dir, epochs=EPOCHS_DEFAULT, patience=PATIENCE_DEFAULT):
    """A run is done when it hit the epoch cap or early-stopped on patience."""
    import json
    sp = os.path.join(run_dir, "state.json")
    if not os.path.exists(sp):
        return False
    with open(sp) as f:
        st = json.load(f)
    return st["last_epoch"] >= epochs - 1 or st["last_epoch"] - st["best_epoch"] >= patience


def stage_prep(args):
    """Disk-budgeted build order for /kaggle/working (~20 GB cap):
    train waveforms (3.3G) -> train mels (0.8G) -> aug pool (6.7G) [peak 10.8G]
    -> DELETE train waveforms -> dev shards (direct from flac) -> eval shards."""
    from src.dataset import build_waveforms, build_original_mels, build_aug_pool, build_clean_mel_shards

    if not os.path.exists("data/processed/train_waveforms.npy"):
        print("=== waveforms train ===")
        build_waveforms("train", workers=4)
    if not os.path.exists("data/processed/train_mels.npy"):
        print("=== original train mels ===")
        build_original_mels("train", workers=4)
    if not os.path.exists("data/processed/train_pool.npy"):
        print("=== augmentation pool (K=8) ===")
        build_aug_pool("train", workers=4)
    if not os.path.exists("data/processed/mel_stats.npz"):
        print("=== global mel stats ===")
        subprocess.run([sys.executable, "scripts/compute_global_stats.py"], check=True)
    # train waveforms are only needed to build the pool — free the 3.3 GB
    if os.path.exists("data/processed/train_pool.npy"):
        os.remove("data/processed/train_waveforms.npy")
        print("removed intermediate train_waveforms.npy")
    for split in ("dev", "eval"):
        shard = f"data/processed/{split}_mel_shards"
        if not (os.path.isdir(shard) and os.listdir(shard)):
            print(f"=== clean mel shards {split} (direct from flac) ===")
            build_clean_mel_shards(split, workers=4)
    print("PREP DONE — promote /kaggle/working/repo/data/processed to a Kaggle "
          "dataset (dvd-processed) and attach it to the training kernels")


def stage_train(args):
    from src.train_kaggle import train_run

    train_run(args.mode, args.seed, epochs=args.epochs, time_budget_h=args.time_budget_h)


def stage_grid(args):
    """Run every grid entry that has no state.json yet, within the time budget.

    If the processed artifacts (augmentation pool etc.) don't exist yet — e.g.
    the prep session hasn't run or its output wasn't attached — run prep first;
    it is idempotent and resumable."""
    if not os.path.exists("data/processed/train_pool.npy"):
        print("=== grid: processed artifacts missing, running prep first ===")
        stage_prep(args)
    from src.train_kaggle import train_run

    # stop cleanly before Kaggle kills the session (~9-12 h GPU cap)
    budget = args.time_budget_h if args.time_budget_h is not None else 8.5
    for mode, seed in GRID_PRIORITY:
        run_dir = os.path.join("/kaggle/working/models", f"dvd_{mode}_seed{seed}")
        if run_finished(run_dir, epochs=args.epochs):
            continue
        print(f"=== grid: {mode} seed {seed} ===")
        train_run(mode, seed, epochs=args.epochs, time_budget_h=budget)
    print("GRID STAGE COMPLETE (or time budget reached — re-commit to continue)")


def stage_eval(args):
    import glob
    import json
    from src.evaluate import evaluate

    for best in sorted(glob.glob("/kaggle/working/models/*/best.keras")):
        run = os.path.basename(os.path.dirname(best))
        for split in ("dev", "eval"):
            out = evaluate(best, split, "clean", out_root="/kaggle/working/results",
                           bootstrap=(args.bootstrap and split == "eval"))
            print(f"{run} {split}: EER {out['metrics']['eer'] * 100:.2f}% "
                  f"t-DCF {out['metrics']['min_tDCF']:.4f}")
    # copy trained models into working output for download
    print("EVAL STAGE DONE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", nargs="?", choices=["prep", "train", "grid", "eval"],
                        default=None,
                        help="default: grid on a GPU session, prep on CPU — so the same "
                             "script works as both the prep kernel and the training kernel")
    parser.add_argument("--mode", default="full")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--time_budget_h", type=float, default=None)
    parser.add_argument("--bootstrap", action="store_true")
    args = parser.parse_args()
    if args.stage is None:
        gpus = []
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices("GPU")
        except Exception:
            pass
        args.stage = "grid" if gpus else "prep"
        # all 15 runs finished -> this push becomes the clean eval sweep
        if args.stage == "grid" and all(
            run_finished(os.path.join("/kaggle/working/models", f"dvd_{m}_seed{s}"),
                         epochs=args.epochs) for m, s in GRID_PRIORITY
        ):
            args.stage = "eval"
        print(f"no stage given: GPU={bool(gpus)} -> stage={args.stage}")
    dict(prep=stage_prep, train=stage_train, grid=stage_grid, eval=stage_eval)[args.stage](args)
