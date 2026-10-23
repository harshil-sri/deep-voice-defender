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

# Wire the ASVspoof data: prefer an attached dataset, else the repo's own copy
KAGGLE_DATA = None
for root in ["/kaggle/input/asvspoof2019-la", "/kaggle/input/asvspoof-2019-la"]:
    if os.path.isdir(root):
        KAGGLE_DATA = root
        break
if KAGGLE_DATA:
    link = os.path.join(REPO, "data", "raw", "LA", "LA")
    os.makedirs(os.path.dirname(link), exist_ok=True)
    if not os.path.exists(link):
        os.symlink(KAGGLE_DATA, link)
    print(f"data -> {KAGGLE_DATA}")

# Wire precomputed processed artifacts (from the prep run's promoted output)
for root in os.listdir("/kaggle/input"):
    proc = os.path.join("/kaggle/input", root, "processed")
    if os.path.isdir(proc):
        link = os.path.join(REPO, "data", "processed")
        if not os.path.exists(link):
            os.symlink(proc, link)
        print(f"processed -> {proc}")
        break

GRID_PRIORITY = [(m, s) for s in (42, 123, 456) for m in ("full", "none", "opus", "resample", "noise")]


def stage_prep(args):
    from src.dataset import build_waveforms, build_original_mels, build_aug_pool, build_clean_mel_shards

    for split in ("train", "dev", "eval"):
        wfp = f"data/processed/{split}_waveforms.npy"
        if not os.path.exists(wfp):
            print(f"=== waveforms {split} ===")
            build_waveforms(split, workers=4)
    if not os.path.exists("data/processed/train_mels.npy"):
        print("=== original train mels ===")
        build_original_mels("train", workers=4)
    if not os.path.exists("data/processed/train_pool.npy"):
        print("=== augmentation pool (K=8) ===")
        build_aug_pool("train", workers=4)
    if not os.path.exists("data/processed/mel_stats.npz"):
        print("=== global mel stats ===")
        subprocess.run([sys.executable, "scripts/compute_global_stats.py"], check=True)
    for split in ("dev", "eval"):
        shard = f"data/processed/{split}_mel_shards"
        if not (os.path.isdir(shard) and os.listdir(shard)):
            print(f"=== clean mel shards {split} ===")
            build_clean_mel_shards(split, workers=4)
        # waveform caches are only intermediates — dev+eval are ~12.5 GB, and
        # /kaggle/working has a 20 GB output limit
        wfp = f"data/processed/{split}_waveforms.npy"
        if os.path.exists(wfp):
            os.remove(wfp)
            print(f"removed intermediate {wfp}")
    print("PREP DONE — promote /kaggle/working/repo/data/processed to a Kaggle "
          "dataset (dvd-processed) and attach it to the training kernels")


def stage_train(args):
    from src.train_kaggle import train_run

    train_run(args.mode, args.seed, epochs=args.epochs, time_budget_h=args.time_budget_h)


def stage_grid(args):
    """Run every grid entry that has no state.json yet, within the time budget."""
    from src.train_kaggle import train_run

    for mode, seed in GRID_PRIORITY:
        state = os.path.join("/kaggle/working/models", f"dvd_{mode}_seed{seed}", "state.json")
        if os.path.exists(state):
            continue
        print(f"=== grid: {mode} seed {seed} ===")
        train_run(mode, seed, epochs=args.epochs, time_budget_h=args.time_budget_h)
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
    parser.add_argument("stage", choices=["prep", "train", "grid", "eval"])
    parser.add_argument("--mode", default="full")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--time_budget_h", type=float, default=None)
    parser.add_argument("--bootstrap", action="store_true")
    args = parser.parse_args()
    dict(prep=stage_prep, train=stage_train, grid=stage_grid, eval=stage_eval)[args.stage](args)
