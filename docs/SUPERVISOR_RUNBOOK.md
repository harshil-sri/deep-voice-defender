# SUPERVISOR RUNBOOK — DeepVoice Defender pipeline

**Audience:** an AI agent (or human) supervising the Kaggle training/eval pipeline.
This file is self-contained. If anything here conflicts with `WORKLOG.md` (top section),
**WORKLOG wins** — it is the live log; this file is the stable playbook.

**Last verified against reality:** 2026-10-25 (session-2 incident diagnosed, seeding bug
fixed, pipeline paused by Harshil — nothing is running on Kaggle).

---

## 1. Mission (why this pipeline exists)

Turn the DeepVoice Defender mini-project into a conference paper: *"How much does
codec-aware augmentation help lightweight audio deepfake detectors?"* — an empirical
robustness study on ASVspoof 2019 LA.

- **Grid:** 5 augmentation modes (`full`, `none`, `opus`, `resample`, `noise`) × 3 seeds
  (42, 123, 456) = **15 training runs**, each ≤60 epochs with early stopping (patience 10).
- **Eval sweep (the immediate next step):** score every finished model on dev+eval
  (clean) → 30 JSONs in `results/`.
- **After that:** codec re-encode matrix on 2019 eval, ASVspoof 2021 LA cross-corpus
  eval, figures (`scripts/make_figures.py`), efficiency numbers (`scripts/efficiency.py`),
  then the paper (`paper/PLAN.md` is the outline).

A run is **finished** when its `state.json` shows `last_epoch >= 59` OR
`last_epoch - best_epoch >= 10`. All 15 runs finished on 2026-10-24 (session 1).

## 2. Golden rules — never violate these

1. **NEVER print, commit, or transmit any Kaggle token/secret contents.** Tokens live at
   `~/.kaggle/access_token` (harshilsrivastav) and
   `~/.kaggle-accounts/mrigankrana/access_token`. Reading them into a command variable is
   fine (`TOKEN=$(cat …)`); echoing them is not. Never commit them.
2. **Never commit model binaries** (`*.keras`) or the 1 GB+ session downloads to GitHub.
   `.gitignore` already covers them — don't force-add.
3. **Push any kernel at most ONCE per supervision run.** If a push fails, diagnose before
   any second push. If the failure message contains "quota" or "GPU", log it in WORKLOG
   and do NOT retry (GPU quota is weekly, resets ~Saturday UTC).
4. **Never interrupt a RUNNING Kaggle session.** No cancel mechanism is worth the risk;
   sessions self-terminate (GPU budget 8.5 h set in the script, Kaggle hard cap 9–12 h).
5. **Never push GPU kernels to mrigankrana** — that account has NO GPU and NO internet
   (phone unverified; CPU-only sessions work). Everything runs as harshilsrivastav.
6. **Never download >~500 MB per supervision run**, EXCEPT training-session outputs
   (models+state, ~1.7 GB) which are the point of the download. NEVER download the prep
   kernel output (~11 GB) — there is no reason to.
7. **Don't re-derive state — read it.** WORKLOG top section first, then this file.
8. When a poll finds nothing to do, **report one line and stop**. Do not invent work.
9. Clean up `/tmp/dvd-*` and `/tmp/*push*` staging dirs when done.
10. **Stop and escalate to Harshil** (write a loud WORKLOG entry + final report, then end)
    if: the same error class appears twice in a row and is not understood; GPU quota is
    exhausted; the grid-state dataset is missing/corrupted; or eval results contradict
    sanity (e.g. every model ≈ 50% EER on dev).

## 3. Accounts, auth, and cheat sheet

| Thing | Value |
|---|---|
| Working account | `harshilsrivastav` — phone-verified: GPU + internet OK |
| Default CLI env | plain `kaggle …` (token at `~/.kaggle/access_token`) |
| mrigankrana env | `KAGGLE_API_TOKEN=$HOME/.kaggle-accounts/mrigankrana/access_token HOME=$HOME/.kaggle-accounts/mrigankrana/.isolated_home KAGGLE_CONFIG_DIR=$HOME/.kaggle-accounts/mrigankrana kaggle …` (CPU/no-internet only; currently unused) |

Key objects (all under harshilsrivastav):

| Object | Role |
|---|---|
| kernel `deepvoice-defender-prep` | CPU; builds mel shards + aug pool into `repo/data/processed`. **COMPLETE — treat as frozen.** Public. |
| kernel `deepvoice-defender-training` | GPU; the workhorse. Auto-picks stage: grid → eval. Repo metadata: `kaggle/kernel-metadata-training-harshil.json` |
| dataset `deep-voice-defender-code` | the repo (code only); mounted as input so no internet is needed |
| dataset `dvd-grid-state-h` | session continuity: `models/` + `results/` uploaded between GPU sessions |
| dataset `anishsarkar22/asvpoof-2019-dataset-la` | public ASVspoof 2019 LA mirror; attached to both kernels |

Status / output / push commands (all DEFAULT env):

```bash
kaggle kernels status harshilsrivastav/deepvoice-defender-training
kaggle kernels output harshilsrivastav/deepvoice-defender-training -p /tmp/out   # only when COMPLETE/ERROR
kaggle datasets status harshilsrivastav/dvd-grid-state-h
kaggle datasets version -p <staging-dir> --dir-mode zip -m "session N"           # upload grid state
```

**Push recipe for the training kernel** (metadata file MUST be named `kernel-metadata.json`):

```bash
rm -rf /tmp/train-push && mkdir -p /tmp/train-push
cp kaggle/deepvoice_training.py /tmp/train-push/
cp kaggle/kernel-metadata-training-harshil.json /tmp/train-push/kernel-metadata.json
kaggle kernels push -p /tmp/train-push        # DEFAULT env = harshilsrivastav
date -u                                        # record this push timestamp in WORKLOG
```

**Fetch a finished/errored session's log WITHOUT downloading output** (works only AFTER
the session ends; while RUNNING the log is empty — do not retry it):

```bash
TOKEN=$(cat ~/.kaggle/access_token)
curl -s -H "Authorization: Bearer $TOKEN" \
  "https://www.kaggle.com/api/v1/kernels/output?user_name=harshilsrivastav&kernel_slug=deepvoice-defender-training" \
  | python3 -c "import json,sys; d=json.load(sys.stdin); [print(e.get('data') or '', end='') for e in json.loads(d.get('log') or '[]')]"
```

## 4. THE NEXT ACTION — run the clean-eval session

Precondition (verified 2026-10-25): all 15 runs finished (session-1 models are in
`dvd-grid-state-h` v2 and locally in `results/kaggle_sessions/1/models`); the seeding bug
is fixed in `kaggle/deepvoice_training.py` (skip prep-output placeholders).

1. `kaggle kernels status …` — must not be RUNNING. (It is COMPLETE from the wasted
   session-2 run; that's fine.)
2. Push the training kernel once (recipe above) → this is **version 3**.
3. Wait ~10 min, then check status: it should be RUNNING (past the wiring phase).
4. **Verify the fix took effect** once the session ends (or immediately from the log if
   it errors): the log must contain BOTH of these lines —
   - `SKIPPED seeding from /kaggle/input/deepvoice-defender-prep/repo/models (prep-output placeholder)` (and the results one)
   - `grid state after seeding: 15/15 runs finished`
   If it instead says `stage=grid` and starts training `dvd_full_seed42` again → the
   seeding wipe happened AGAIN → **STOP, escalate** (do not let it burn another session).
5. Expected runtime ~45–90 min (15 models × dev+eval; eval split is 71 k utterances).
   When COMPLETE → download output (≈1.7 GB, mostly `models/` re-upload; the new part is
   `results/`):
   ```bash
   kaggle kernels output harshilsrivastav/deepvoice-defender-training -p results/kaggle_sessions/3
   ```
6. **Verify eval completeness:** `results/kaggle_sessions/3/results/eval/` must contain
   30 JSONs named `{run}__{split}__{condition}.json` (15 runs × dev+eval, condition
   `clean` — pattern from `src/evaluate.py`). If fewer, see §6.
7. Upload the updated state (models unchanged + results/) as version 3 of
   `dvd-grid-state-h`:
   ```bash
   rm -rf /tmp/dvd-grid-v3 && mkdir -p /tmp/dvd-grid-v3
   cp -r results/kaggle_sessions/3/models /tmp/dvd-grid-v3/
   cp -r results/kaggle_sessions/3/results /tmp/dvd-grid-v3/
   cat > /tmp/dvd-grid-v3/dataset-metadata.json <<'EOF'
   {"title":"dvd-grid-state-h","id":"harshilsrivastav/dvd-grid-state-h","licenses":[{"name":"CC0-1.0"}]}
   EOF
   kaggle datasets version -p /tmp/dvd-grid-v3 --dir-mode zip -m "session 3: clean eval sweep"
   kaggle datasets status harshilsrivastav/dvd-grid-state-h   # wait for "ready"
   rm -rf /tmp/dvd-grid-v3 /tmp/train-push
   ```
8. Aggregate locally — stats.py reads exactly `results/eval/*.json`:
   ```bash
   mkdir -p results/eval
   cp results/kaggle_sessions/3/results/eval/*.json results/eval/
   python3 scripts/stats.py            # writes results/tables/{summary.csv,ablation.tex,…}
   ```
   Put the final per-mode table (mean±std EER and min t-DCF per mode over 3 seeds, dev
   and eval) into WORKLOG. Commit the JSONs + tables to GitHub; never the `.keras` files.

## 5. After the eval sweep — remaining study work (in order)

1. **Codec matrix** on 2019 eval: `scripts/build_codec_matrix.py` builds re-encoded
   variants (Opus/MP3/AAC etc. at several bitrates) of the eval set; then score the 15
   models on each cell. This is THE robustness result of the paper. It can run as another
   Kaggle GPU session (attach the same datasets; adapt `deepvoice_training.py`'s eval
   stage or add a `codec` stage) or locally if disk allows — prefer Kaggle (no local GPU
   trust; WSL disk numbers are NOT real free space).
2. **ASVspoof 2021 LA cross-corpus eval**: `scripts/fetch_asvspoof2021.py` (Zenodo DOI
   10.5281/zenodo.4837263; asvspoof.org keys bundle returned 503 last tried 2026-10-23 —
   retry, or find the Kaggle mirror dataset). Score the 15 models with
   `src/protocol_2021.py`. NOTE: 2021 eval has no official labels — report EER on the
   known subset / use the 2021 scoring conventions documented in `docs/asvspoof2021.md`.
3. **Figures**: `scripts/make_figures.py` (DET/ROC/boxplots per mode). Needs the eval
   JSONs from §4.
4. **Efficiency**: `scripts/efficiency.py` (params/MACs/latency table for the paper).
5. **Paper**: follow `paper/PLAN.md`. `paper/references.bib` entries are UNVERIFIED —
   verify every citation against arXiv/actual venues before including.

## 6. Failure playbooks

- **Session ERROR:** fetch the log (§3 recipe). Known classes:
  - *OOM* → reduce batch size in `src/train_kaggle.py`, commit, push ONCE.
  - *Disk* → prune `/kaggle/working` (e.g. skip `ckpt_latest.keras` saves), push ONCE.
  - *Quota/GPU at push time* → log in WORKLOG, wait for weekly reset, do NOT retry.
  - *Empty log + no output files* → infra kill (happened to prep v2); just push ONCE more.
- **Retrain-wipe (the session-2 bug):** symptom = log shows `stage=grid` and per-run
  epoch lines for runs that were already finished. Cause = models/ seeded from a
  placeholder dir. Fixed 2026-10-25 (seeding skips `deepvoice-defender-prep` paths).
  If it ever recurs: STOP, escalate; canonical models remain in `dvd-grid-state-h` +
  `results/kaggle_sessions/1/models`.
- **Eval incomplete (fewer than 30 JSONs):** stage_eval is NOT resumable (it re-scores
  everything), but a full pass is only ~45–90 min — just push the kernel again. Note:
  the auto-stage goes to eval only when all 15 are finished AND a GPU is present.
- **`nvidia-smi` missing / cuInit error in log:** the session had no GPU. On
  harshilsrivastav this means quota exhaustion or a Kaggle hiccup — check the push
  response and WORKLOG for quota notes; wait if quota.
- **CLI "Metadata file not found":** the staged metadata must be named exactly
  `kernel-metadata.json`.
- **"not valid kernel sources"** on push: the kernel_source must be COMPLETE and
  same-owner (or public). Prep is both; don't change this without reading WORKLOG.

## 7. Known facts you should not re-derive (burned time learning these)

- Kaggle mounts datasets at `/kaggle/input/datasets/<owner>/<slug>/…` (new nesting);
  the kernel script globs 1–4 levels deep — trust it.
- Kaggle `--dir-mode zip` uploads zips of directories and Kaggle unpacks them on ingest.
- Kernel logs are only fetchable after the session ends; while RUNNING, status is the
  only signal. A healthy grid session = still RUNNING after 10 min.
- GPU training is nondeterministic across sessions even at fixed seed (session-1 vs
  session-2 dev EERs differ by up to ~37 points on one run). Report within-session
  seed spread; never mix sessions' models in one statistic.
- mrigankrana: no GPU, no internet (phone unverified). Blocked for training until
  verified; his `dvd-grid-state` + training kernel are unused.
- GPU quota: 30 h/week per account. Session 1 used ~5.2 h, session 2 ~8 h.
- The 500 MB download guard: session outputs (models+results) are the exception.

## 8. Current numbers snapshot (session 1, canonical; dev = clean ASVspoof 2019 LA)

| mode | seed 42 | seed 123 | seed 456 | mean dev EER |
|---|---|---|---|---|
| full | 3.61% | 4.00% | 2.51% | **3.37%** |
| none | 6.14% | 8.75% | 5.53% | 6.80% |
| opus | 8.87% | 9.54% | 8.87% | 9.09% |
| resample | 14.80% | 17.74% | 11.58% | 14.71% |
| noise | 22.02% | 3.06% | 28.66% | 17.91% |

Dev is CLEAN, so these say "augmentation costs clean-dev fit" — the robustness claim
comes from the eval sweep + codec matrix. `full` wins on dev; `noise` has huge seed
variance. No eval-split numbers exist yet (that is §4's job).
