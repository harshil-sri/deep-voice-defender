"""Aggregate evaluation results into the paper's tables and figures data.

Reads:
  results/eval/{run}__{split}__{condition}.json   (from src/evaluate.py)
  results/scores/*.csv                            (for paired tests / CIs)

Writes:
  results/tables/summary.csv        every (run, split, condition) with EER/t-DCF
  results/tables/ablation.tex       main ablation table: mode x seed + mean+-std (clean eval)
  results/tables/codec_matrix.tex   codec-robustness matrix: mode x condition, mean EER
  results/tables/per_attack.tex     per-attack EER table (best mode vs none)

Usage:  python scripts/stats.py [--split eval] [--condition clean] [--out results/tables]
"""

import os
import sys
import json
import glob
import argparse
import re
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RUN_RE = re.compile(r"dvd_(?P<mode>none|noise|resample|opus|full)_seed(?P<seed>\d+)")
MODES = ["none", "noise", "resample", "opus", "full"]


def load_eval_jsons(out_root):
    rows = []
    for path in sorted(glob.glob(os.path.join(out_root, "eval", "*.json"))):
        with open(path) as f:
            d = json.load(f)
        m = RUN_RE.match(d["run"])
        rows.append({
            "run": d["run"],
            "mode": m.group("mode") if m else d["run"],
            "seed": int(m.group("seed")) if m else -1,
            "split": d["split"],
            "condition": d["condition"],
            "eer": d["metrics"]["eer"],
            "min_tdcf": d["metrics"]["min_tDCF"],
            "per_attack": d["metrics"].get("per_attack", {}),
            "n": d.get("num_samples"),
        })
    return rows


def fmt_pct(x):
    return f"{x * 100:.2f}"


def write_summary_csv(rows, out_dir):
    path = os.path.join(out_dir, "summary.csv")
    with open(path, "w") as f:
        f.write("run,mode,seed,split,condition,eer_percent,min_tdcf,n\n")
        for r in rows:
            f.write(f"{r['run']},{r['mode']},{r['seed']},{r['split']},{r['condition']},"
                    f"{r['eer'] * 100:.4f},{r['min_tdcf']:.4f},{r['n']}\n")
    print(f"wrote {path}")


def ablation_table(rows, split, condition, out_dir):
    """mode x seed -> EER (+ min t-DCF for the mean), LaTeX booktabs."""
    by_mode = defaultdict(dict)
    tdcf = defaultdict(dict)
    for r in rows:
        if r["split"] == split and r["condition"] == condition:
            by_mode[r["mode"]][r["seed"]] = r["eer"]
            tdcf[r["mode"]][r["seed"]] = r["min_tdcf"]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Ablation of augmentation strategies on ASVspoof 2019 LA "
        f"({split}, {condition}). EER (\\%) and minimum t-DCF, three seeds.}}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{lccc|cc}",
        r"\toprule",
        r"Augmentation & \multicolumn{3}{c|}{EER (\%)} & min t-DCF & EER mean$\pm$std (\%) \\",
        r"strategy & s=42 & s=123 & s=456 & (mean) & \\" ,
        r"\midrule",
    ]
    for mode in MODES:
        if mode not in by_mode:
            continue
        seeds = sorted(by_mode[mode])
        vals = [by_mode[mode][s] * 100 for s in seeds]
        cells = " & ".join(fmt_pct(by_mode[mode][s]) for s in seeds) if len(seeds) == 3 else \
                " & ".join(fmt_pct(by_mode[mode][s]) for s in seeds) + " & " * (3 - len(seeds))
        t = np.mean([tdcf[mode][s] for s in seeds])
        lines.append(f"{mode} & {cells} & {t:.3f} & "
                     f"{np.mean(vals):.2f}$\\pm${np.std(vals):.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    path = os.path.join(out_dir, "ablation.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {path}")


def codec_matrix_table(rows, split, out_dir, metric="eer"):
    """mode x condition -> mean EER across seeds."""
    conds = sorted({r["condition"] for r in rows if r["split"] == split})
    cell = defaultdict(dict)
    for r in rows:
        if r["split"] == split:
            cell[(r["mode"], r["condition"])].setdefault("eers", []).append(r[metric])
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Robustness of augmentation strategies to codec degradation: "
        "EER (\\%, mean over seeds) on ASVspoof 2019 LA "
        f"{split} re-encoded through real codecs.}}",
        r"\label{tab:codec_matrix}",
        r"\begin{tabular}{l" + "c" * len(conds) + "}",
        r"\toprule",
        "Augmentation & " + " & ".join(conds) + r" \\",
        r"\midrule",
    ]
    for mode in MODES:
        if not any((mode, c) in cell for c in conds):
            continue
        vals = []
        for c in conds:
            e = cell.get((mode, c))
            vals.append(fmt_pct(np.mean(e["eers"])) if e else "--")
        lines.append(mode + " & " + " & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    path = os.path.join(out_dir, "codec_matrix.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {path}")


def per_attack_table(rows, split, condition, out_dir):
    """Per-attack EER: mean across seeds per mode (attacks as columns)."""
    by_mode = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["split"] == split and r["condition"] == condition and r["per_attack"]:
            for atk, d in r["per_attack"].items():
                by_mode[mode_ := r["mode"]][atk].append(d["eer"])
    attacks = sorted({a for m in by_mode for a in by_mode[m]})
    lines = [
        r"\begin{table*}[t]", r"\centering",
        r"\caption{Per-attack EER (\%, mean over seeds) on ASVspoof 2019 LA "
        f"{split}, {condition}. A07--A19 are unseen attacks.}}",
        r"\label{tab:per_attack}",
        r"\begin{tabular}{l" + "c" * len(attacks) + "}",
        r"\toprule",
        "Augmentation & " + " & ".join(attacks) + r" \\",
        r"\midrule",
    ]
    for mode in MODES:
        if mode not in by_mode:
            continue
        vals = [fmt_pct(np.mean(by_mode[mode].get(a, [np.nan]))) for a in attacks]
        lines.append(mode + " & " + " & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    path = os.path.join(out_dir, "per_attack.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_root", default="results")
    parser.add_argument("--split", default="eval")
    parser.add_argument("--condition", default="clean")
    args = parser.parse_args()
    out_dir = os.path.join(args.out_root, "tables")
    os.makedirs(out_dir, exist_ok=True)
    rows = load_eval_jsons(args.out_root)
    if not rows:
        sys.exit("no eval jsons found — run src/evaluate.py first")
    write_summary_csv(rows, out_dir)
    ablation_table(rows, args.split, args.condition, out_dir)
    codec_matrix_table(rows, args.split, out_dir)
    per_attack_table(rows, args.split, args.condition, out_dir)
