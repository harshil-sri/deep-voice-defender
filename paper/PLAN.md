# Paper Plan — Claims → Experiments → Tables/Figures

**Working title:** *How Much Does Codec-Aware Augmentation Help Lightweight Audio Deepfake Detection? An Empirical Study on Codec-Degraded and Telecom Speech*

**Target venues (in order of fit):** IEEE Signal Processing Letters (fast, 4 pages,
rolling) → Interspeech 2027 (~March 2027 deadline, the natural home) → DCASE
Workshop 2027 / IEEE SLT. Write to full-paper length; cut to SPL length if late.

## Claims to defend

- **C1 (headline):** Training with true codec augmentation substantially
  improves a lightweight CNN countermeasure's robustness to codec-degraded and
  telecom speech, at negligible parameter cost.
  → Table `codec_matrix.tex` (mode × codec-condition EER), ASVspoof 2021 LA
  cross-corpus table.
- **C2 (ablation):** Which augmentation matters. True Opus vs naive resampling
  (S1) is the most distinctive comparison — naive resampling is what most
  "we simulated codecs" papers actually do.
  → Table `ablation.tex` (mode × seed, mean±std, clean eval) + codec matrix
  columns opus_6k/12k/24k/48k.
- **C3 (protocol rigor):** Official metrics: EER + official min t-DCF computed
  with the provided ASV scores at the ASV's EER operating point; model selection
  on dev (contains unknown attacks); 3 seeds; bootstrap CIs.
  → Numbers in every table; CI column in the ablation table.
- **C4 (analysis):** Per-attack breakdown (known A01–A06 vs unseen A07–A19;
  TTS vs VC families).
  → Table `per_attack.tex`; training curves figure.
- **C5 (efficiency):** Lightweight angle: params (~X M), MACs, real-time factor,
  inference latency (CPU + GPU).
  → Efficiency table (generate from a small script; TODO: scripts/efficiency.py).

## Run matrix (15 training runs)

| mode | what it is | seeds |
|---|---|---|
| none | raw audio | 42, 123, 456 |
| noise | Gaussian @ SNR U(5,20) dB | 42, 123, 456 |
| resample | naive 16k→8k→16k (v1's "opus") | 42, 123, 456 |
| opus | TRUE libopus @ {6,12,24,48} kbps | 42, 123, 456 |
| full | 50% none / 25% opus / 25% noise | 42, 123, 456 |

Priority order: all seed 42 first, then 123, then 456. Grid priority list is in
`kaggle/deepvoice_training.py` (GRID_PRIORITY).

## Evaluation matrix

- Splits: dev (model selection only) and eval (all tables).
- Conditions on eval: `clean` + 11 codec conditions from `src/codec.CODEC_MATRIX`
  (opus 6/12/24/48k, mp3 16/32/64k, aac 16/32k, amr-nb 4.75/12.2k).
- Cross-corpus: ASVspoof 2021 LA eval (Zenodo 10.5281/zenodo.4817532; Kaggle
  mirror exists). Verify 2021 eval labels + protocol format at download time;
  report EER there (t-DCF needs 2021 ASV scores — check availability; else EER
  only, footnote it).

## Figures

1. Training curves: dev EER vs epoch for the five modes (seed 42) — from
   `models/*/log.csv`.
2. Codec robustness line plot: EER vs Opus bitrate per mode (subset of the matrix).
3. DET curves (clean eval): best mode vs none.
4. (Optional) Architecture diagram of the CNN + augmentation pool pipeline.

All figures generated from `results/` only — no hand-drawn numbers. TODO:
`scripts/make_figures.py`.

## Writing checklist (before submission)

- [ ] Verify every references.bib entry against the actual paper.
- [ ] Verify published baseline numbers quoted for context (2019 GMM/LSTM
      baselines, AASIST, RawBoost) — pull from the actual papers, not memory.
- [ ] State the static-pool design honestly (K=8 fixed degradations per
      utterance) and justify (compute); it's a real limitation to acknowledge.
- [ ] Report the v1 recipe's informal 11.95% EER only as motivation, never in
      results tables.
- [ ] Reproducibility statement: exact commands, seeds, Kaggle hardware, repo URL.
- [ ] Ethical statement: dataset licenses (ASVspoof ODC-By), defensive intent.
