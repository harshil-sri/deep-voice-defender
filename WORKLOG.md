# WORKLOG — DeepVoice Defender → Conference Paper

**Purpose:** Running handoff log. If this session hits limits, ANY model can resume seamlessly by reading this file top-to-bottom (most recent work at the TOP), plus `PROJECT_CONTEXT.md` and `paper/PLAN.md`.

**Paper direction (user-approved, 2026-10-23):**
- Story: *"How much does codec-aware augmentation help lightweight audio deepfake detectors?"* — empirical robustness study, NOT a SOTA-accuracy claim.
- Scope: **Core grid + S1 only** = 5 augmentation modes × 3 seeds (42/123/456) = **15 training runs**:
  - `none` (no aug), `noise` (Gaussian @ random SNR 5–20 dB), `opus` (TRUE Opus via ffmpeg, random bitrate {6,12,24,48}k), `resample` (naive 16k→8k→16k — S1 head-to-head vs `opus`), `full` (50% none / 25% opus / 25% noise).
- Preprocessing: **global-stat normalization**. All 15 runs trained fresh on Kaggle; v1 numbers are informal reference only.
- **WhatsApp mini-set DROPPED.** Cross-corpus = ASVspoof 2021 LA eval (Zenodo DOI 10.5281/zenodo.4837263, Kaggle mirror exists) + codec-re-encode matrix on 2019 eval.
- All GPU work on Kaggle (user has multiple account tokens, 30 GPU-h/wk each) and/or Colab. NO local training (local TF broken — protobuf mismatch).

---

## Current status — BUILD COMPLETE, WAITING ON KAGGLE SETUP

### DONE (2026-10-23, all committed to git main)
- **Metrics:** `src/metrics.py` implements OFFICIAL min t-DCF from the ASVspoof 2019 evaluation plan pp. 8–10 (read from `data/raw/asvspoof2019_evaluation_plan.pdf`). Uses provided ASV scores (`data/raw/LA/LA/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.{dev,eval}.gi.trl.scores.txt`, line-aligned with ASV protocol in `..._asv_protocols/`). Validated: trivial CM → exactly 1.0, perfect CM → 0, bootstrap CIs. Plus EER, per-attack EER (`src/protocol.py` has loaders; protocol format `SPEAKER UTID - ATTACK LABEL`, ASV score lines `ATTACK KEY SCORE`).
- **Preprocessing:** `src/preprocessing.py` — 65024-sample fixed length → mel (128,128) exact (128 mels, hop 512, n_fft 1024); `top_db=80` clamp (silent audio would give −inf); global-stat norm via `data/processed/mel_stats.npz` (train mean ≈ −0.898, std ≈ 7.91 dB). Legacy min-max kept for app.py.
- **True codecs:** `src/codec.py` — Opus/MP3/AAC/AMR-NB ffmpeg round-trips; CODEC_MATRIX = 11 eval conditions. NOTE: ffmpeg muxer must be inferred from extension (.ogg/.mp3/.m4a/.amr).
- **Augmentation:** `src/augmentation.py` — none/noise/resample/opus/full, seeded rng.
- **Data pipeline:** `src/dataset.py` — waveform caches (float16 memmap), STATIC AUGMENTATION POOL (`train_pool.npy` (N,8,128,128): 3 opus + 2 resample + 3 noise mels per utterance; true-opus on-the-fly is ~100 ms/utt → 45 min/epoch, hence static pool), tf.data datasets via `_make_fetch` (verified: none-mode output matches recomputed pipeline to 0.004 = float16 quantization), clean-mel shard builders for dev/eval/codec-matrix. All builders chunk-resumable (`--start/--end`).
- **Trainer:** `src/train_kaggle.py` — per-epoch ckpt + state.json + log.csv, resume across session kills, model selection on DEV EER (patience 10), `--time_budget_h` clean stops, class_weights flag (default off).
- **Evaluator:** `src/evaluate.py` — writes `results/scores/{run}__{split}__{condition}.csv` (utt_id,label,attack,score — SOURCE OF TRUTH) + `results/eval/*.json` with official metrics.
- **Scripts:** `scripts/stats.py` (summary.csv + ablation/codec_matrix/per_attack .tex), `scripts/compute_global_stats.py`, `scripts/build_codec_matrix.py` (resumable), `scripts/make_figures.py` (training curves, DET, opus-bitrate plot — tested on synthetic data), `scripts/efficiency.py` (params/MACs/latency; model ≈ 0.16 GMACs; run on Kaggle), `scripts/kaggle_push.py`, `scripts/fetch_asvspoof2021.py` (see below).
- **Launchers:** `kaggle/deepvoice_training.py` (stages: prep [CPU ~3-4h] / train / grid [priority: all seed42 first] / eval) + `kaggle/kernel-metadata.json` (TODO: replace YOUR_KAGGLE_USERNAME) + `colab/deepvoice_training.ipynb`.
- **Paper:** `paper/main.tex` (IEEEtran skeleton), `paper/references.bib` (seeded, UNVERIFIED), `paper/PLAN.md` (claims→tables map, venue strategy: SPL → Interspeech 2027 → DCASE/SLT).
- **Docs:** PROJECT_CONTEXT.md rewritten; this file.
- **Tests:** `tests/sanity_check.py` — ALL PASS locally (metrics, mel shapes, 4 codec round-trips, 5 aug modes, opus timing 104 ms/utt).
- Real-data smoke test passed locally: waveform cache → mels → pool (120 rows) → global stats → dev shards (2000 rows) → fetch verified for all 5 modes.

### Bugs found & fixed during smoke testing (learn from these)
1. ffmpeg can't infer muxer from `.bin` → codec container extensions required.
2. `per_attack_eer` indexing bug (fixed).
3. Silent audio → −inf log-mels → `top_db=80` + nan_to_num clamp added.
4. float16 accumulation overflow in np.std over big arrays → cast float32 first (compute_global_stats).

### NEXT STEPS (in order — user is setting up a spare Kaggle account now)
1. **User:** provide Kaggle token(s): `~/.kaggle/kaggle.json` (or per-account jsons for `scripts/kaggle_push.py --kaggle_json`).
2. Upload data: `python scripts/kaggle_push.py dataset` (~7.5 GB from data/raw/LA/LA). Fix the `"id"` in the generated dataset-metadata.json after creation. OR attach a public Kaggle ASVspoof-2019-LA dataset after verifying counts (train 25380 / dev 24841 / eval 71237 flac).
3. Make repo reachable from Kaggle: push code as a dataset or make the GitHub repo public (kaggle kernel clones YOUR_USERNAME/Deep-Voice-Defender by default).
4. Push kernel (edit `kaggle/kernel-metadata.json`), run `prep` on a **CPU** session (~3–4 h, resumable). Then promote `repo/data/processed` output → new dataset `dvd-processed`, attach to training kernel metadata.
5. Sanity training: `grid` with seed-42 `full` first — verify epoch time (target 1–3 min/epoch on P100) and dev EER curve before burning quota.
6. Full grid (15 runs × ~60 epochs, est. 30–45 GPU-h total). Sessions: `grid --time_budget_h 8`, re-commit to continue.
7. Eval sweeps: `eval` stage (clean dev+eval), then codec matrix (`scripts/build_codec_matrix.py` ~5–6 h CPU for eval, resumable) + `src/evaluate.py --condition ...` per model.
8. Cross-corpus: `python scripts/fetch_asvspoof2021.py` (see `docs/asvspoof2021.md` for VERIFIED facts: LA eval = Zenodo record 4837263; labels + official ASV scores only from asvspoof.org LA-keys-full.tar.gz — server was 503 on 2026-10-23, HF `SpeechAntiSpoofingBenchmarks/ASVspoof2021_LA` labels.parquet is a verified fallback; 2021 protocol = single-col trl list, labels/codec/attack in keys bundle `trial_metadata.txt` 8 columns; Kaggle mirror mohammedabdeldayem/avsspoof-2021 has the full LA eval audio), evaluate all 15 models on it.
9. `scripts/stats.py` + `scripts/make_figures.py` + `scripts/efficiency.py` → paper tables/figures; write paper per `paper/PLAN.md` checklist.

### KEY NUMBERS / FACTS FOR HANDOFF
- v1 reference (informal, NOT for paper tables): EER 11.95%, acc 87.14%; its min t-DCF 0.0100 is INVALID (CM-only approximation).
- Official 2019 baselines to verify when writing: LFCC-GMM eval EER ≈ 8.09% / min t-DCF ≈ 0.2116; LFCC-LSTM ≈ 9.66% / ≈ 0.1936; AASIST ≈ 0.83% EER. Our CNN is not SOTA — contribution is the robustness ablation + rigor. Model ≈ 0.16 GMACs (lightweight story).
- t-DCF LA constants: π_tar=0.9405, π_non=0.0095, π_spoof=0.05, C_miss_asv=1, C_fa_asv=10, C_miss_cm=1, C_fa_cm=10; ASV threshold at its own EER (target vs nontarget); normalize by min(C1,C2).
- Mel spec: 128 mels, hop 512, n_fft 1024, 65024 samples → (128,128); model input (128,128,1); stats file data/processed/mel_stats.npz.
- Score convention: sigmoid, HIGH = bonafide; label 1 = bonafide.
- Kaggle limits: ~9–12 h GPU sessions, 20 GB working-output cap (prep deletes dev/eval waveform caches to fit; processed artifacts ≈ 14 GB), 30 GPU-h/wk/account.
- Local TF import BROKEN (protobuf). Local testing = numpy/scipy/librosa only (tests/sanity_check.py).

### GOTCHAS
- `results/` holds v1-era eval json for `deep_voice_defender_v1_fixed` — analysis scripts skip it (no dvd_ prefix); keep as legacy reference.
- Old `src/train.py` is the legacy local trainer — do NOT use for paper runs.
- `kaggle/kernel-metadata.json` and colab notebook have YOUR_USERNAME placeholders to fill.
- Never quote a min t-DCF that didn't come from `src/metrics.py`.
