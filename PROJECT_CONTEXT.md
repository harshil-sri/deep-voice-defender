# Deep-Voice Defender — Project Context

**Last Updated:** 2026-10-23
**Status:** v2 research infrastructure COMPLETE (not yet trained). Paper-grade rewrite in progress.
**Read next:** `WORKLOG.md` (handoff log — most recent state), `paper/PLAN.md` (claims → experiments map).

---

## 1. What this project is now

DeepVoice Defender started as a college mini-project: a 3-block CNN on 128×128
log-mel spectrograms detecting audio deepfake speech (bonafide vs spoof) on
ASVspoof 2019 LA. v1 achieved EER 11.95% / accuracy 87.14% on the eval set
(its old "99%" README claim and its min t-DCF 0.0100 were invalid — the t-DCF
was computed without the required ASV scores).

The project is being upgraded into a conference-grade **empirical study**:

> *"How much does codec-aware augmentation help lightweight audio deepfake
> detection? An empirical study on codec-degraded and telecom speech."*

User-approved scope (2026-10-23):
- **Core grid + S1 = 15 training runs:** 5 augmentation modes × 3 seeds:
  - `none`, `noise` (SNR 5–20 dB), `opus` (**true libopus** via ffmpeg),
    `resample` (naive 16k→8k→16k — v1's fake "opus", kept for the S1
    head-to-head), `full` (50% none / 25% opus / 25% noise). Seeds 42/123/456.
- **Global-stat normalization** (per-utterance min-max retired).
- **No WhatsApp mini-set**; cross-corpus = ASVspoof 2021 LA eval + a local
  codec-re-encode matrix on 2019 eval/dev.
- **All training on Kaggle** (30 GPU-h/wk/account) or Colab. Never train locally
  (local TF is broken: protobuf gencode/runtime mismatch; and the 4060 is off-limits).

## 2. Architecture of the v2 pipeline

```
ASVspoof 2019 LA flac
  │  src/dataset.py build-waveforms   (float16 memmap, 65024 smp ≈ 4.065 s)
  ▼
waveform caches ──► src/dataset.py build-pool ──► STATIC AUGMENTATION POOL
                  build-mels                     train_pool.npy float16 (N,8,128,128)
                  scripts/compute_global_stats   3 opus + 2 resample + 3 noise mels/file
                                                 (true-opus is ~100 ms/utt via ffmpeg,
                                                  too slow on-the-fly → static pool)
  ▼
src/train_kaggle.py   tf.data ← pool lookups only; per-epoch ckpt; resume-safe;
                      model selection on DEV EER; time-budgeted clean stops
  ▼
models/dvd_{mode}_seed{seed}/best.keras + log.csv + state.json
  ▼
src/evaluate.py       → results/scores/{run}__{split}__{condition}.csv (raw scores!)
                        results/eval/{run}__{split}__{condition}.json
  ▼
scripts/stats.py      → results/tables/{ablation,codec_matrix,per_attack}.tex
scripts/make_figures.py, scripts/efficiency.py → figures + efficiency table
```

Key implementation facts (verified by tests/sanity_check.py):
- Waveform length **65024** samples → mel (128 mels, hop 512, n_fft 1024) is
  **exactly (128,128)**; model input (128,128,1).
- **Official min t-DCF** implemented in `src/metrics.py` from the ASVspoof 2019
  evaluation plan (pp. 8–10): t-DCF = C1·P_miss + C2·P_fa, normalized by
  min(C1,C2), ASV at its own EER point on target-vs-nontarget, using the
  provided ASV scores in `data/raw/LA/LA/ASVspoof2019_LA_asv_scores/`. The
  v1 number (0.0100) is invalid and must never be quoted.
- True codec simulation in `src/codec.py` (Opus/MP3/AAC/AMR-NB round-trips via
  ffmpeg; binary from `$FFMPEG_BIN` → PATH → imageio_ffmpeg).
- `log-mel` uses `top_db=80` clamp (silent frames otherwise → −inf).
- Any float16 reduction must cast to float32 first (np.std overflow bug found
  in practice).
- Score convention everywhere: **high = bonafide**, label 1 = bonafide.

## 3. Run/eval protocol

Grid priority (see GRID_PRIORITY in kaggle/deepvoice_training.py): all seed-42
runs first, then 123, then 456 — a complete single-seed story exists early.
Evaluation conditions: `clean` + 11 codec re-encodes of eval
(opus 6/12/24/48k, mp3 16/32/64k, aac 16/32k, amr-nb 4.75/12.2k —
`src/codec.CODEC_MATRIX`) + ASVspoof 2021 LA eval (cross-corpus).

Kaggle workflow (kaggle/deepvoice_training.py stages):
1. `prep` (CPU session, ~3–4 h): waveform caches → mels → pool → stats → dev/eval
   shards. Promote `/kaggle/working/repo/data/processed` to dataset `dvd-processed`.
2. `grid` (GPU sessions, time-budgeted): trains remaining runs, resumable.
3. `eval`: scores all finished models on dev+eval clean.
Codec matrix + 2021 runs are follow-up sessions (scripts/build_codec_matrix.py
is resumable; scripts/fetch_asvspoof2021.py downloads 2021 LA).

## 4. Current state / next actions

- [x] v2 pipeline built and smoke-tested locally (data path verified on real
      ASVspoof files; metrics validated on synthetic data)
- [ ] **User:** put a Kaggle token at ~/.kaggle/kaggle.json (multiple accounts
      supported via scripts/kaggle_push.py --kaggle_json)
- [ ] Upload data dataset (`python scripts/kaggle_push.py dataset`), push kernel,
      run `prep` on a CPU session, run `grid` on GPU sessions
- [ ] Evaluate + stats + figures; then 2021 LA cross-corpus
- [ ] Paper: paper/main.tex skeleton exists; fill from generated tables
      (paper/PLAN.md has the claims→tables map and writing checklist)

## 5. Hard-won constraints (don't re-litigate)

| Decision | Choice | Why |
|---|---|---|
| Training hardware | Kaggle P100/T4 + Colab only | user's accounts; local TF broken; 4060 off-limits |
| Augmentation delivery | static pool K=8 | true-opus ≈ 100 ms/utt → 45 min/epoch on-the-fly; pool → ~1–2 min/epoch |
| Model selection | dev EER, patience 10 | ASVspoof protocol; v1's train-loss early stop was wrong |
| Class weights | off (v1 recipe continuity) | one variable at a time; flag exists |
| Mel normalization | global stats (train mean −0.898, std 7.91 dB) | min-max destroys level info |
| min t-DCF | official implementation only | v1's 0.0100 was CM-only approximation |
| Eval artifacts | raw scores CSVs are the source of truth | all analysis rebuildable from them |

## 6. File map (v2 additions)

```
src/protocol.py        2019 protocol + ASV trial loaders
src/metrics.py         EER, official min t-DCF, per-attack, bootstrap
src/preprocessing.py   65024-sample mels, global-stat norm (legacy min-max kept for app.py)
src/codec.py           true codec round-trips + CODEC_MATRIX conditions
src/augmentation.py    5 modes, seeded
src/dataset.py         caches/pool/shards + tf.data
src/train_kaggle.py    resumable trainer, dev-EER selection
src/evaluate.py        scores CSV + metrics JSON
scripts/               stats, global-stats, codec-matrix, figures, efficiency, kaggle push, 2021 fetch
kaggle/                kernel script + metadata
colab/                 notebook
paper/                 main.tex, references.bib, PLAN.md
tests/sanity_check.py  no-TF sanity suite (run: python3 tests/sanity_check.py)
WORKLOG.md             handoff log — read this first when resuming
```

v1 files kept as-is: app.py, src/{model,fix_model,inference,predict}.py,
src/train.py (legacy local trainer — do not use for the paper).
