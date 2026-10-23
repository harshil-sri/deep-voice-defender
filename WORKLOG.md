# WORKLOG — DeepVoice Defender → Conference Paper

**Purpose:** Running handoff log. If this session hits limits, ANY model can resume seamlessly by reading this file top-to-bottom (most recent work at the TOP), plus `PROJECT_CONTEXT.md` and `paper/PLAN.md`.

**Paper direction (user-approved, 2026-10-23):**
- Story: *"How much does codec-aware augmentation help lightweight audio deepfake detectors?"* — empirical robustness study, NOT a SOTA-accuracy claim.
- Scope: **Core grid + S1 only** = 5 augmentation modes × 3 seeds (42/123/456) = **15 training runs**:
  - `none` (no aug), `noise` (Gaussian @ random SNR 5–20 dB), `opus` (TRUE Opus codec via ffmpeg, random bitrate), `resample` (naive 16k→8k→16k — the S1 head-to-head vs `opus`), `full` (50% none / 25% true-opus / 25% noise).
- Preprocessing: **global-stat normalization** (replaces per-utterance min-max). All 15 runs retrained on Kaggle; the old local `none_seed42` run and old results are ABANDONED (v1 numbers kept only as informal reference).
- **WhatsApp mini-set DROPPED.** Cross-corpus = ASVspoof 2021 LA eval (Zenodo DOI 10.5281/zenodo.4817532, has Kaggle mirror) + local codec-re-encode matrix on 2019 eval/dev.
- All GPU work on Kaggle (user has multiple account tokens, 30 GPU-h/wk each) and/or Colab. NO local training (RTX 4060 reserved; local TF is broken anyway — protobuf mismatch).

---

## Current status — what is done / what remains

### DONE (engineering build-out, 2026-10-23)
- Read official t-DCF definition from `data/raw/asvspoof2019_evaluation_plan.pdf` pp. 8–10. Implemented in `src/metrics.py` (see file header for the exact equations).
- `src/metrics.py` — EER (ROCCH-friendly), official min t-DCF (uses provided ASV scores in `data/raw/LA/LA/ASVspoof2019_LA_asv_scores/`), per-attack EER, bootstrap CI.
- `src/preprocessing.py` v2 — 65024-sample fixed length → mel (128×128), global-stat norm via `data/processed/mel_stats.npz` (built by `scripts/compute_global_stats.py`). Old v1 API (`audio_to_mel` min-max) kept for `app.py` compat.
- `src/codec.py` — true codec simulation through ffmpeg subprocess (Opus/MP3/AAC/AMR-NB at multiple bitrates). Binary discovery: `$FFMPEG_BIN` → `shutil.which` → `imageio_ffmpeg`.
- `src/augmentation.py` v2 — modes: none/noise/resample/opus/full, all seeded.
- `src/dataset.py` — float16 waveform cache (train ≈ 3.3 GB, fits Kaggle RAM) + tf.data pipeline with parallel aug+mel.
- `src/train_kaggle.py` — CLI trainer, per-epoch checkpoint + CSVLogger, **resume across Kaggle 12 h session kills**, early stopping on **dev EER** (precomputed dev mels).
- `src/evaluate.py` v2 — emits per-utterance raw scores CSV (utt_id, label, attack, score) + metrics JSON with official metrics.
- `scripts/compute_global_stats.py`, `scripts/build_codec_matrix.py`, `scripts/precompute_mels.py`, `scripts/stats.py` (LaTeX table generation), `scripts/kaggle_push.py`.
- `kaggle/` notebook + `colab/` notebook.
- `paper/` — LaTeX skeleton, references.bib, `paper/PLAN.md` (claims → table/figure map).
- `tests/sanity_check.py` — EER/t-DCF/codec/mel sanity tests, all passing locally.

### NEXT STEPS (in order)
1. **User action:** provide Kaggle API token(s) (put `kaggle.json` at `~/.kaggle/kaggle.json` or set `KAGGLE_USERNAME`/`KAGGLE_KEY`), review `WORKLOG.md` + `paper/PLAN.md`.
2. Push ASVspoof 2019 LA as private Kaggle dataset: `python scripts/kaggle_push.py dataset` (local data is only ~7.5 GB). OR attach an existing public Kaggle ASVspoof-2019-LA dataset after verifying file counts (train 25380 / dev 24841 / eval 71237 flac).
3. On Kaggle: run one sanity training (full, seed 42, short epochs) → verify epoch time (target 2–6 min/epoch on P100) and quota burn before launching everything.
4. Launch core grid in priority order: all seed-42 runs first (none, full, opus, resample, noise), then 123, then 456. Each run: `python src/train_kaggle.py --aug_mode X --seed Y` inside the Kaggle notebook (time-budgeted per commit session).
5. Eval sweeps AFTER grid: build codec matrix + precompute mels on Kaggle (GPU/CPU), then `src/evaluate.py` per model × condition → `results/scores/*.csv`.
6. Cross-corpus: download ASVspoof 2021 LA eval (Zenodo / Kaggle mirror), verify eval labels are available (check the Zenodo record; 2021 protocol format differs slightly — adapt `src/evaluate.py --protocol-format 2021` if needed), run all 15 models on it.
7. `scripts/stats.py` → tables; `paper/` writing.

### KEY NUMBERS / FACTS FOR HANDOFF
- Baseline (v1 model, min-max norm, WRONG min t-DCF): EER 11.95%, acc 87.14%. Reference rows in paper must use OFFICIAL min t-DCF from `src/metrics.py`.
- ASVspoof 2019 LA official baselines (verify exact values from papers when writing): LFCC-GMM eval EER ≈ 8.09% / min t-DCF ≈ 0.2116; LFCC-LSTM ≈ 9.66% / ≈ 0.1936. AASIST eval EER ≈ 0.83%. Our model is NOT SOTA — the paper's contribution is the codec-robustness ablation, lightweight angle, and cross-corpus analysis. Own the low accuracy honestly.
- t-DCF LA constants: π_tar=0.9405, π_non=0.0095, π_spoof=0.05, C_miss_asv=1, C_fa_asv=10, C_miss_cm=1, C_fa_cm=10. ASV threshold = ASV's own EER point on target-vs-nontarget. Normalized by min(C1, C2).
- Data paths (local & Kaggle mount): `data/raw/LA/LA/ASVspoof2019_LA_{train,dev,eval}/flac`, protocols at `data/raw/LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.{train.trn,dev.trl,eval.trl}.txt`. Protocol cols: VOKID / UT T-ID? — actually format = `speaker uttid - attack label` (see `src/protocol.py`). ASV scores: `data/raw/LA/LA/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.{dev,eval}.gi.trl.scores.txt`.
- Score convention: sigmoid output, HIGH = bonafide. Label 1 = bonafide, 0 = spoof (both in protocol loader and metrics).
- Mel spec: 128 mels, hop 512, n_fft 1024, waveform fixed at 65024 samples → exactly (128, 128) spectrogram; model input (128, 128, 1).
- Local TF import is BROKEN (protobuf gencode 6.31.1 vs runtime 5.29.6). Do NOT try to fix unless needed — training is on Kaggle. Local tests use numpy/scipy/librosa only.
- git: work happens on `main`; user asked for frequent commits.

### GOTCHAS
- `results/` currently holds v1 results — keep as `results_v1_legacy` reference only, never quote in the paper.
- Kaggle session limits: commit runs ~9–12 h GPU. Notebook trains in time-budgeted batches with per-epoch checkpoints → resumable.
- ffmpeg on Kaggle is preinstalled; locally we use `imageio_ffmpeg` static binary.
- `src/train.py` (old local trainer) is legacy; use `src/train_kaggle.py`.
