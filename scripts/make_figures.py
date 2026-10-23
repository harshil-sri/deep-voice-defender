"""Paper figures: training curves, DET curves on clean eval, Opus-bitrate robustness.

Reads:
  models/dvd_{mode}_seed42/log.csv              (training curves, seed 42 only)
  results/scores/{run}__{split}__{condition}.csv (DET curves, score high = bonafide)
  results/eval/{run}__{split}__{condition}.json  (codec robustness, mean over seeds)

Writes:
  results/figures/training_curves.png
  results/figures/det_clean.png
  results/figures/opus_bitrate.png

Missing inputs are skipped gracefully (printed, not fatal).

Usage:  python scripts/make_figures.py [--models_dir models] [--results_dir results]
        [--out_dir results/figures] [--split eval] [--det_runs dvd_full_seed42 dvd_none_seed42]
"""

import os
import sys
import csv
import json
import glob
import argparse
import re
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # for stats.RUN_RE
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # for src.metrics

from stats import RUN_RE, MODES
from src.metrics import compute_eer

MODE_LABELS = {
    "none": "No augmentation",
    "noise": "Additive noise",
    "resample": "Resampling",
    "opus": "Opus codec",
    "full": "Full pipeline",
}
MODE_COLORS = {
    "none": "#7f7f7f",
    "noise": "#1f77b4",
    "resample": "#2ca02c",
    "opus": "#ff7f0e",
    "full": "#d62728",
}
OPUS_CONDITIONS = ["opus_6k", "opus_12k", "opus_24k", "opus_48k"]


def training_curves(models_dir, out_dir):
    """Dev EER vs epoch for the five modes (seed 42)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    plotted = skipped = 0
    for mode in MODES:
        path = os.path.join(models_dir, f"dvd_{mode}_seed42", "log.csv")
        if not os.path.exists(path):
            print(f"skipped (no log): {path}")
            skipped += 1
            continue
        epochs, eers = [], []
        with open(path) as f:
            for row in csv.DictReader(f):
                epochs.append(int(row["epoch"]))
                eers.append(float(row["dev_eer"]) * 100)
        order = np.argsort(epochs)
        ax.plot(np.array(epochs)[order], np.array(eers)[order],
                label=MODE_LABELS.get(mode, mode), color=MODE_COLORS.get(mode), lw=1.8)
        plotted += 1
    if not plotted:
        plt.close(fig)
        print("training curves: nothing to plot")
        return
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Dev set EER (%)")
    ax.set_title("Training curves (seed 42)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "training_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def det_curves(results_dir, split, det_runs, out_dir):
    """DET curves (FPR vs FNR, log axes) on the clean condition."""
    fig, ax = plt.subplots(figsize=(5, 5))
    plotted = 0
    for run in det_runs:
        path = os.path.join(results_dir, "scores", f"{run}__{split}__clean.csv")
        if not os.path.exists(path):
            print(f"skipped (no scores): {path}")
            continue
        labels, scores = [], []
        with open(path) as f:
            for row in csv.DictReader(f):
                labels.append(int(row["label"]))
                scores.append(float(row["score"]))
        labels = np.asarray(labels)
        scores = np.asarray(scores)
        pos, neg = scores[labels == 1], scores[labels == 0]
        if len(pos) == 0 or len(neg) == 0:
            print(f"skipped (single class): {path}")
            continue
        eer, _ = compute_eer(pos, neg)
        fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
        fnr = 1.0 - tpr
        # Log axes cannot show exact 0/1 corner points; drop them.
        keep = (fpr > 0) & (fnr > 0)
        ax.plot(fpr[keep], fnr[keep], label=f"{MODE_LABELS.get(_mode(run), run)} (EER {eer * 100:.2f}%)",
                color=MODE_COLORS.get(_mode(run)), lw=1.8)
        plotted += 1
    if not plotted:
        plt.close(fig)
        print("DET curves: nothing to plot")
        return
    lims = [1e-4, 1.0]
    ax.plot(lims, lims, "k--", lw=0.8, alpha=0.5)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("False positive rate (spoof accepted)")
    ax.set_ylabel("False negative rate (bonafide rejected)")
    ax.set_title(f"DET curves ({split}, clean)")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.grid(which="both", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "det_clean.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def _mode(run):
    m = RUN_RE.match(run)
    return m.group("mode") if m else run


def opus_bitrate(results_dir, split, out_dir):
    """EER vs Opus bitrate per mode (mean over seeds), from eval jsons."""
    # (mode, condition) -> [eer per seed]
    cell = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(results_dir, "eval", "*.json"))):
        with open(path) as f:
            d = json.load(f)
        m = RUN_RE.match(d.get("run", ""))
        if not m:  # e.g. v1-era results
            print(f"skipped (not a dvd run): {path}")
            continue
        if d.get("split") != split or d.get("condition") not in ["clean"] + OPUS_CONDITIONS:
            continue
        cell[(m.group("mode"), d["condition"])].append(d["metrics"]["eer"])

    conds = []
    if any(c == "clean" for _, c in cell):
        conds.append(("clean", "clean"))
    conds += [(c, c.replace("opus_", "")) for c in OPUS_CONDITIONS]
    fig, ax = plt.subplots(figsize=(6, 4))
    plotted_modes = set()
    for mode in MODES:
        xs, ys = [], []
        for cond, tick in conds:
            eers = cell.get((mode, cond))
            if eers:
                xs.append(tick)
                ys.append(float(np.mean(eers)) * 100)
                plotted_modes.add(mode)
        if xs:
            ax.plot(range(len(xs)), ys, "o-", label=MODE_LABELS.get(mode, mode),
                    color=MODE_COLORS.get(mode), lw=1.8)
    if not plotted_modes:
        plt.close(fig)
        print("opus bitrate: nothing to plot")
        return
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([t for _, t in conds])
    ax.set_xlabel("Opus bitrate (kbps)")
    ax.set_ylabel("EER (%)")
    ax.set_title(f"Codec robustness ({split}, mean over seeds)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "opus_bitrate.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models_dir", default="models")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--out_dir", default="results/figures")
    parser.add_argument("--split", default="eval")
    parser.add_argument("--det_runs", nargs="+", default=["dvd_full_seed42", "dvd_none_seed42"],
                        help="runs for the DET curves (clean condition)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    training_curves(args.models_dir, args.out_dir)
    det_curves(args.results_dir, args.split, args.det_runs, args.out_dir)
    opus_bitrate(args.results_dir, args.split, args.out_dir)
