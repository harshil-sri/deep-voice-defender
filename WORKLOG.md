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

## Current status — PREP COMPLETE; TRAINING GRID SESSION 1 RUNNING (mrigankrana GPU)

### Session 1 started (2026-10-24 13:21 UTC / 18:51 IST)
- mrigank prep v2 COMPLETE at ~13:21 UTC (pushed 06:30:52 UTC, ~6 h 50 m total — the
  tail is the eval-shard phase). Output verified via v1 API: 70 files incl.
  `repo/data/processed/{dev,eval}_mel_shards/shard_*.npz` (kernel_sources will mount
  this at `/kaggle/input/deepvoice-defender-prep/...`).
- **mrigankrana/deepvoice-defender-training v2 pushed 13:21:58 UTC** — GPU session,
  8.5 h budget, internet off, code dataset + corpus + grid-state + prep-output mounts.
  Confirmed RUNNING 13:23 UTC. It seeds /kaggle/working/{models,results} from
  dvd-grid-state (empty), then runs the grid in priority order
  (seeds 42 first: full/none/opus/resample/noise) and stops cleanly at the budget.
- Expected: ~2–3 runs finished this session (each run ≈ 2.5–3 h at 60 epochs on
  Kaggle's P100/T4x2). Next polls: when COMPLETE → download output (≤500 MB),
  grid-progress entry, upload dvd-grid-state version, re-push to continue.
- Legacy note: harshilsrivastav prep v8 (public rerun) still running on his CPU — ignore.

### Account consolidation + internet elimination (2026-10-24, 06:20–06:35 UTC, user-directed)
- Prep v7 on harshilsrivastav COMPLETE (~6 h) — but the first training push revealed
  **the prep kernel had reverted to PRIVATE** (root cause: `"is_private": "true"` string
  in `kernel-metadata-prep.json` — every CLI re-push re-enforced private; anonymous page
  returned 404, and mrigankrana got `kernels.get denied`). kernel_sources requires the
  source to be public OR same-owner.
- **User decision: run EVERYTHING under mrigankrana** so privacy never matters.
- mrigank prep v1 then failed in 40 s: `Could not resolve host: github.com` —
  **mrigankrana's sessions have NO internet** (account phone-verification was never
  confirmed; Kaggle gates internet on it). The script's git-clone fallback died.
- **Fix — kernels are now fully self-contained (enable_internet: false):**
  1. Created dataset `mrigankrana/deep-voice-defender-code` (ready, private, same-owner,
     ~300 KB: whole repo minus data/models/results/logs/.git/binaries; --dir-mode zip,
     Kaggle unpacks directory zips on ingest — verified via `kaggle datasets files`).
  2. `kaggle/deepvoice_training.py` REPO wiring reworked: `INPUT_ROOTS`/`_input_globs`
     moved above it; it now searches ALL mount depths for a dir containing
     `src/train_kaggle.py` (matches `deep-voice-defender-code` dataset) and copies it to
     /kaggle/working/repo; git clone remains only as a fallback.
  3. Both mrigank kernels: dataset_sources += deep-voice-defender-code, internet OFF.
     `kernel-metadata-prep.json` is now the **mrigank** prep metadata (harshil prep is legacy).
- **mrigank prep v2 RUNNING** (pushed 06:30:52 UTC, ETA ~6 h ≈ 12:30 UTC / 18:00 IST).
  Confirmed RUNNING at 06:33 UTC — outlived v1's 40 s failure, so repo wiring worked.
- Training kernel v1 (pushed 06:21 UTC with internet ON + invalid prep source) is a
  wasted fast-fail session — expected. The automation pushes training v2 (same-owner
  prep source, code dataset, internet off) once mrigank prep v2 COMPLETES.
- harshilsrivastav/deepvoice-defender-prep v8 (public rerun) still burning his CPU
  quota harmlessly — IGNORE it.

### Prep incident + resolution (2026-10-24, 04:00–06:20 IST)
- v2 (ran 21:12–~23:50 UTC) died mid-pool-build with an EMPTY log and NO output files —
  log purged, cause unknown. Since nothing persisted, prep restarted from scratch.
- v3–v6 then failed FAST with clear logs. Root cause: **Kaggle changed input mounting** —
  datasets now mount at `/kaggle/input/datasets/<owner>/<slug>/…` (confirmed via
  in-kernel listing: `kaggle inputs: ['datasets']` → `datasets/anishsarkar22/asvpoof-2019-dataset-la`).
  The old hardcoded mount paths matched nothing.
- Fix (committed): `kaggle/deepvoice_training.py` now locates the corpus/processed/
  models/results by globbing `/kaggle/input` and `/kaggle/input/datasets` at 1–4 levels
  (`_input_globs()` helper, keyed on the unique `ASVspoof2019_LA_cm_protocols` dir),
  prints the input tree for debugging, and FATALs with a clear message if the corpus
  is missing. Also fixed a `_glob` import-order bug introduced en route (v4).
- **v7 RUNNING** (pushed ~06:15 IST). Log/API diagnostics recipe that worked:
  `GET https://www.kaggle.com/api/v1/kernels/output?user_name=…&kernel_slug=…`
  with `Authorization: Bearer <access_token>` → JSON with `log` (list of stream entries)
  and `files` — no download needed.

## Previous status — FULLY AUTONOMOUS (no user steps at all); prep still RUNNING, training starts automatically when it completes

### Fully hands-off design (final, 2026-10-24 ~03:20 IST)
- **No browser dataset promotion is needed.** The training kernel mounts the prep
  kernel's output directly via `kernel_sources: ["harshilsrivastav/deepvoice-defender-prep"]`
  in `kaggle/kernel-metadata.json` — Kaggle mounts a kernel source's last successful
  output at `/kaggle/input/<slug>/` (here `/kaggle/input/deepvoice-defender-prep/repo/data/processed/...`),
  which the kernel's deep wiring glob picks up. Works cross-account because the prep
  kernel is PUBLIC. (The `dvd-processed` dataset idea is ABANDONED — don't create it.)
- Continuity across training sessions: `mrigankrana/dvd-grid-state` dataset (created,
  status ready, private, same-owner with the kernel — uploads via mrigankrana's access
  token, verified). The kernel seeds `/kaggle/working/{models,results}` from any
  `/kaggle/input/*/models|results` at startup.
- ZCode cron automation "DeepVoice pipeline monitor" (every 30 min) runs the whole
  state machine: prep status → push training kernel once prep is COMPLETE → on each
  finished/failed session: download small output → archive to dvd-grid-state → re-push
  to continue the grid → auto-eval after all 15 runs → stats + commit. User can sleep.
- Prep started 21:12 UTC (02:42 IST); 3–4 h ⇒ COMPLETE ≈ 05:45–06:45 IST.
- mrigankrana GPU quota is fresh (30 h/wk). Grid ≈ 20–25 GPU-h + eval ⇒ should fit one
  week; automation records quota errors and stops retrying until weekly reset.

## Previous status — AUTONOMOUS PIPELINE ARMED (cron automation polls every 30 min); prep still RUNNING

### Automation (created 2026-10-23 late night) — how the pipeline runs itself
- A ZCode cron automation (every 30 min) drives everything else: when prep completes it
  **notifies the user to do the ONE-TIME browser step** (Output tab → New Dataset →
  title `dvd-processed`, PUBLIC — user preference: no 10 GB download through the local
  disk, and WSL's free-disk number is not real disk), then pushes the GPU training
  kernel under mrigankrana, then on every poll: monitors the session, downloads the
  (small, few hundred MB) finished output, uploads it as a new version of
  **`mrigankrana/dvd-grid-state`** (placeholder created, status `ready`), and re-pushes
  the kernel to continue the grid until all 15 runs + clean eval are done.
- `dvd-processed` is created by the USER as **harshilsrivastav** and made **PUBLIC**
  (public ⇒ attachable by mrigankrana's kernel). Metadata sources:
  `anishsarkar22/asvpoof-2019-dataset-la`, `harshilsrivastav/dvd-processed`,
  `mrigankrana/dvd-grid-state`.
- Kernel script changes enabling this (committed): deep processed wiring
  (`/kaggle/input/*/repo/data/processed` etc.), seeds `/kaggle/working/{models,results}`
  from the grid-state dataset, grid now runs unfinished runs only (epoch-cap/patience
  check via `run_finished()`), 8.5 h clean time budget, and auto-stage: GPU session with
  all 15 runs finished → runs the clean dev+eval sweep automatically.
- User already made the prep kernel PUBLIC (helps: CLI can fetch its output without
  owner-only auth). If the user happens to click "New Dataset" in the Output tab first,
  the automation detects `dvd-processed` exists and skips its own creation (nested
  `repo/data/processed` layout is handled by the deep wiring).
- `kaggle/kernel-metadata.json` dataset_sources now: `anishsarkar22/asvpoof-2019-dataset-la`,
  `harshilsrivastav/dvd-processed`, `harshilsrivastav/dvd-grid-state`.
- Push command (mrigankrana): `python scripts/kaggle_push.py kernel --kaggle_json ~/.kaggle-accounts/mrigankrana/kaggle.json --kernel_dir kaggle`
- Progress readouts: `kaggle kernels status harshilsrivastav/deepvoice-defender-prep` /
  `kaggle kernels status mrigankrana/deepvoice-defender-training` (as mrigankrana env, see below);
  run-to-run progress in WORKLOG "grid progress" section below (automation appends).

## Previous status — PREP RUNNING (harshilsrivastav, CPU); mrigankrana access token VERIFIED; waiting on prep to start GPU grid

### Auth resolution (final, 2026-10-23 night) — how to run kaggle as mrigankrana
- New-style **access token** for mrigankrana saved at `~/.kaggle-accounts/mrigankrana/access_token` (VERIFIED: `config view` introspects to mrigankrana, auth_method ACCESS_TOKEN).
- CLI lookup order for tokens: `KAGGLE_API_TOKEN` env (a file path works!) → `$HOME/.kaggle/access_token`. **KAGGLE_CONFIG_DIR is never used for tokens.**
- `scripts/kaggle_push.py --kaggle_json ~/.kaggle-accounts/mrigankrana/kaggle.json` now exports KAGGLE_API_TOKEN + isolated HOME automatically. Manual commands:
  `KAGGLE_API_TOKEN=~/.kaggle-accounts/mrigankrana/access_token HOME=~/.kaggle-accounts/mrigankrana/.isolated_home KAGGLE_CONFIG_DIR=~/.kaggle-accounts/mrigankrana kaggle ...`
- Legacy kaggle.json keys CANNOT upload (datasets/kernels) — reads only.
- harshilsrivastav: GPU quota exhausted this week (CPU fine, prep running). GPU quota resets weekly.

### Sequence to start training (in order)
1. `kaggle kernels status harshilsrivastav/deepvoice-defender-prep` → wait for COMPLETE (~3–4 h from ~21:35 local).
2. In browser as **harshilsrivastav**: open the prep kernel → Output → create dataset from `repo/data/processed` → name it **dvd-processed**.
3. Edit `kaggle/kernel-metadata.json`: dataset_sources = ["anishsarkar22/asvpoof-2019-dataset-la", "harshilsrivastav/dvd-processed"].
4. Push GPU training kernel (grid auto-runs): `python scripts/kaggle_push.py kernel --kaggle_json ~/.kaggle-accounts/mrigankrana/kaggle.json --kernel_dir kaggle`
5. Monitor: `KAGGLE_API_TOKEN=~/.kaggle-accounts/mrigankrana/access_token HOME=~/.kaggle-accounts/mrigankrana/.isolated_home KAGGLE_CONFIG_DIR=~/.kaggle-accounts/mrigankrana kaggle kernels status mrigankrana/deepvoice-defender-training`
6. Each finished session: download /kaggle/working output (models + state), re-commit the kernel to continue the grid.

### Kaggle auth findings (2026-10-23 evening) — IMPORTANT
- mrigankrana's **legacy kaggle.json key** authenticates fine for READS (`datasets list/files`, `kernels status`) but the CLI (2.2.x) **rejects it for all UPLOADS** (datasets create, kernels push) — requires OAuth or a new-style access token.
- Harshil's default `~/.kaggle/access_token` IS a new-style token → default CLI runs as **harshilsrivastav** and CAN push. His **GPU quota (30h) is already exhausted this week**; CPU quota is separate and prep uses CPU.
- **Prep kernel v2 is RUNNING**: https://www.kaggle.com/code/harshilsrivastav/deepvoice-defender-prep (CPU session, ~3–4 h, resumable). v1 was killed and superseded because of a disk-budget bug (fixed: dev/eval now decode straight to mel shards; no 9 GB waveform caches; peak ~11 GB).
- Data source: public Kaggle mirror **anishsarkar22/asvpoof-2019-dataset-la** (official zip layout, corpus under `…/LA/`; the kernel symlinks `/kaggle/input/asvpoof-2019-dataset-la/LA` → `data/raw/LA/LA`). The private 7.5 GB upload was abandoned (CLI can't upload with legacy key). Two mirrors are byte-identical to the official zip (anishsarkar22, hahunavth).
- Training kernel metadata (`kaggle/kernel-metadata.json`) still points at mrigankrana + GPU. **To unblock GPU training, pick one:**
  1. mrigankrana generates a new-style token (kaggle.com → Settings → API → "Create New Token") and we save it to `~/.kaggle-accounts/mrigankrana/access_token`, or
  2. run `kaggle auth login` (OAuth device flow) with the mrigankrana config dir, or
  3. in the browser as mrigankrana, create a kernel manually (attach `anishsarkar22/asvpoof-2019-dataset-la`, enable GPU+Internet, paste `kaggle/deepvoice_training.py`), or
  4. wait for harshilsrivastav's weekly GPU quota reset and push the training kernel under his account (`sed mrigankrana/harshilsrivastav` on the metadata).
- **After prep completes (check `kaggle kernels status harshilsrivastav/deepvoice-defender-prep`):** promote its output `repo/data/processed` to a dataset named `dvd-processed` (web UI: kernel → Output → "New Dataset"), then add `harshilsrivastav/dvd-processed` to the training kernel's `dataset_sources` and push the GPU training kernel (grid stage runs automatically on GPU).

### Kaggle auth (resolved 2026-10-23) — READ BEFORE RUNNING ANY KAGGLE COMMAND
- Token for **mrigankrana** lives at `~/.kaggle-accounts/mrigankrana/kaggle.json` (legacy API key, VERIFIED: resolves via `kaggle config view` + authenticated `datasets list --mine`).
- **QUIRK:** kaggle CLI 2.2.x IGNORES `KAGGLE_CONFIG_DIR` when `~/.kaggle/access_token` exists (Harshil's own tokens live there → commands silently run as harshilsrivastav). Env vars KAGGLE_USERNAME/KAGGLE_KEY do NOT override it; upgrading to 2.2.5 does not fix it.
- **Fix applied:** `scripts/kaggle_push.py --kaggle_json ...` runs the CLI with an isolated HOME → correct account automatically.
- **For manual kaggle commands as mrigankrana, ALWAYS prefix both vars:**
  `HOME=/home/harshil/.kaggle-accounts/mrigankrana/.isolated_home KAGGLE_CONFIG_DIR=~/.kaggle-accounts/mrigankrana kaggle ...`
- Local protobuf restored to 7.x after a kaggle-CLI upgrade downgraded it — **TF now imports locally (CPU)** and a micro end-to-end train (tf.data batch for all 5 modes → model step, 3.3M params → dev-EER path) was verified. Local GPU still unavailable (fine — Kaggle trains).
- Dev protocol quirk: bonafide trials are the FIRST 2548 rows of the dev protocol, spoofs after — never slice dev into a single block and expect both classes (compute_eer now raises on empty classes instead of returning NaN).

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
