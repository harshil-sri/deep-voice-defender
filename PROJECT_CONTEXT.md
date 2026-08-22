# Deep-Voice Defender - Project Context & Roadmap

**Last Updated:** 2026-09-19  
**Status:** Working end-to-end pipeline, zero evaluation, fake metrics in README  
**Target:** Publishable research paper (Option A: 2-week plan)

---

## 1. Current State of the Project

### What Exists (Working Code)

| File | Lines | Description | Verified |
|------|-------|-------------|----------|
| `app.py` | 220 | Streamlit UI — loads model, plots spectrogram, predicts spoof/bonafide | ✅ Runs |
| `src/preprocessing.py` | 40 | `load_audio()` (resample 16kHz, pad/trim to 4s), `audio_to_mel()` (128 mel bins, min-max normalized) | ✅ Runs |
| `src/augmentation.py` | 26 | `simulate_opus()` (16k→8k→16k resample), `add_noise()` (Gaussian, 0.005 factor), `augment_audio()` (50% none, 25% OPUS, 25% noise) | ✅ Runs |
| `src/model.py` | 47 | 3× Conv2D blocks (32→64→128 filters) + BatchNorm + MaxPool + Dropout → Flatten → Dense(128) → Dense(1, sigmoid) | ✅ Runs |
| `src/train.py` | 160 | `AudioDataGenerator` (Keras `Sequence`), class weights {0:1, 1:9}, EarlyStopping + ReduceLROnPlateau, 40 epochs | ✅ Runs |
| `src/predict.py` | 46 | CLI inference — loads `.keras`, runs same preprocessing, prints verdict | ✅ Runs |
| `deep_voice_defender_v1.keras` | ~3MB | Pre-trained model file | ✅ Exists |
| `requirements.txt` | 18 | TensorFlow[CUDA], librosa, streamlit, etc. | ✅ Installs |
| `data/raw/` | — | ASVspoof 2019 LA dataset (train + eval + protocols) | ✅ Downloaded |

### What Does NOT Exist (Critical Gaps)

| Missing Component | Why It Matters |
|-------------------|----------------|
| **Evaluation script** | No way to compute EER, min t-DCF, accuracy, AUC on eval set |
| **Real metrics** | README claims "99% accuracy" and "Works on WhatsApp" — zero evidence |
| **Ablation study** | Cannot isolate contribution of OPUS vs. noise vs. none |
| **Baseline comparison** | No vanilla CNN (no augmentation) to compare against |
| **Training logs/curves** | No TensorBoard, no loss/accuracy plots saved |
| **Cross-corpus test** | Never tested on actual WhatsApp-compressed audio |
| **Statistical rigor** | Single seed, no confidence intervals, no multiple runs |

### The "Fake Numbers" Problem

**README.md currently claims:**
- "99% accuracy on clean data" → **Unverified, no evaluation code exists**
- "Works on WhatsApp" → **Simulates OPUS in training, but never tested on real WhatsApp audio**
- Comparison table (Standard vs. Defender) → **Purely aspirational, no experimental data**

**Graphify analysis confirmed:** 27% of graph edges are INFERRED (not extracted from code). The inferred edges (main→load_audio, predict_suspect→audio_to_mel, etc.) are **actually correct** — the code does call them. But the performance claims are fabricated.

---

## 2. What Option A Is

**Option A = Real Research Paper (2 weeks)**

A complete, honest, reproducible experimental paper with:
1. **Real evaluation** on ASVspoof 2019 LA eval set (EER, min t-DCF, AUC, accuracy)
2. **Ablation study** — 4 training runs: (a) no augmentation, (b) OPUS only, (c) noise only, (d) full
3. **Baseline comparison** — vanilla CNN vs. Defender CNN
4. **Cross-corpus validation** — test on real WhatsApp/Telegram compressed audio
5. **Statistical rigor** — 3 seeds, confidence intervals, training curves
6. **Written paper** — 6-8 pages: Introduction, Related Work, Method, Experiments, Results, Conclusion

**Target venue:** College research journal / conference (BTEC level)  
**Deliverable:** PDF + GitHub repo with reproducible code + trained models

---

## 3. How We Continue (Workflow)

### Phase 1: Evaluation Infrastructure (Days 1-2)
```
Create src/evaluate.py
    ↓
Run on ASVspoof 2019 LA eval set (already in data/raw/)
    ↓
Get: EER, min t-DCF, AUC, accuracy, ROC/DET curves
    ↓
Save results to results/eval_baseline.json
```

### Phase 2: Ablation Training (Days 3-7)
```
Modify train.py to accept --augmentation_mode flag
    ↓
Train 4 variants (3 seeds each = 12 runs):
  1. none      — no augmentation
  2. opus_only — simulate_opus only
  3. noise_only — add_noise only
  4. full      — current augment_audio (50/25/25)
    ↓
Evaluate all 12 models on eval set
    ↓
Statistical comparison (mean ± std EER)
```

### Phase 3: Cross-Corpus Test (Days 8-9)
```
Collect 50+ real WhatsApp voice notes (bonafide + spoof)
    ↓
Create src/test_whatsapp.py
    ↓
Run all 4 model variants on WhatsApp test set
    ↓
Report degradation vs. clean eval
```

### Phase 4: Paper Writing (Days 10-14)
```
Generate all tables/figures from results/
    ↓
Write paper in Overleaf/LaTeX or Markdown→PDF
    ↓
Sections: Abstract, Intro, Related Work, Method, Experiments, Results, Discussion, Conclusion
    ↓
Push final code + models + paper to GitHub
```

---

## 4. Detailed To-Do List

### ✅ Already Done (Foundation)
- [x] End-to-end pipeline works (train → model → predict → Streamlit)
- [x] ASVspoof 2019 LA dataset downloaded
- [x] Pre-trained model exists (`deep_voice_defender_v1.keras`)
- [x] GPU environment configured (train.py auto-configures LD_LIBRARY_PATH)

### 🔧 Phase 1: Evaluation (Days 1-2)
- [ ] **Create `src/evaluate.py`** — load model, iterate eval protocol, compute EER/min_tDCF/AUC/acc
- [ ] **Add ROC/DET curve plotting** — save to `results/figures/roc.png`, `det.png`
- [ ] **Add confusion matrix** — save to `results/figures/cm.png`
- [ ] **Run baseline evaluation** — current `deep_voice_defender_v1.keras` on eval set
- [ ] **Save metrics** — `results/eval_full_model.json`
- [ ] **Verify "99% accuracy" claim** — replace with real numbers in README

### 🔧 Phase 2: Ablation Study (Days 3-7)
- [ ] **Modify `src/augmentation.py`** — add `mode` parameter: `none`, `opus`, `noise`, `full`
- [ ] **Modify `src/train.py`** — accept `--aug_mode` CLI arg, pass to generator
- [ ] **Create `scripts/run_ablation.sh`** — loop over 4 modes × 3 seeds = 12 trainings
- [ ] **Run 12 trainings** — each saves `deep_voice_defender_{mode}_seed{}.keras`
- [ ] **Evaluate all 12 models** — `results/ablation/raw_scores.csv`
- [ ] **Compute statistics** — mean ± std EER per mode, significance tests
- [ ] **Generate ablation table** — `results/tables/ablation.tex`

### 🔧 Phase 3: Cross-Corpus (Days 8-9)
- [ ] **Collect WhatsApp test set** — 25 bonafide + 25 spoof voice notes (record/collect)
- [ ] **Create `src/test_cross_corpus.py`** — evaluate all 4 best models on WhatsApp set
- [ ] **Report relative degradation** — (EER_whatsapp - EER_clean) / EER_clean
- [ ] **Save results** — `results/cross_corpus.json`

### 🔧 Phase 4: Paper & Polish (Days 10-14)
- [ ] **Training curves** — add TensorBoard callback, export loss/acc/AUC plots
- [ ] **Model complexity table** — params, MACs, inference latency (CPU/GPU)
- [ ] **Write paper** — `paper/main.tex` or `paper/main.md`
- [ ] **Create reproducibility README** — exact commands to reproduce all results
- [ ] **Push to GitHub** — clean repo with `results/`, `paper/`, trained models

---

## 5. Key Technical Decisions (Don't Re-Debate)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Eval metric** | EER + min t-DCF | ASVspoof 2019 LA standard |
| **Ablation modes** | 4 (none, opus, noise, full) | Isolates each component |
| **Seeds** | 3 (42, 123, 456) | Balance rigor vs. compute time |
| **Class weights** | Keep {0:1, 1:9} | ASVspoof LA is 9:1 bonafide:spoof |
| **Input shape** | (128, 128, 1) | Fixed by preprocessing (4s @ 16kHz → 128×128) |
| **Framework** | TensorFlow/Keras | Already used, model exists |
| **Paper format** | LaTeX (Overleaf) | Standard for research |

---

## 6. File Structure After Completion

```
Deep-Voice-Defender/
├── app.py                          # Streamlit UI (unchanged)
├── requirements.txt
├── README.md                       # Updated with REAL results
├── PROJECT_CONTEXT.md              # This file
├── src/
│   ├── __init__.py
│   ├── preprocessing.py            # Unchanged
│   ├── augmentation.py             # + mode parameter
│   ├── model.py                    # Unchanged
│   ├── train.py                    # + --aug_mode CLI
│   ├── predict.py                  # Unchanged
│   ├── evaluate.py                 # NEW
│   └── test_cross_corpus.py        # NEW
├── scripts/
│   └── run_ablation.sh             # NEW
├── data/
│   └── raw/                        # ASVspoof 2019 LA (unchanged)
├── results/
│   ├── eval_full_model.json
│   ├── ablation/
│   │   ├── raw_scores.csv
│   │   └── stats_summary.json
│   ├── cross_corpus.json
│   ├── figures/
│   │   ├── roc.png
│   │   ├── det.png
│   │   ├── cm.png
│   │   └── training_curves/
│   └── tables/
│       └── ablation.tex
├── paper/
│   ├── main.tex
│   ├── references.bib
│   └── figures/                    # Copied from results/figures
└── models/
    ├── deep_voice_defender_v1.keras           # Original
    ├── deep_voice_defender_none_seed42.keras
    ├── deep_voice_defender_opus_seed42.keras
    ├── deep_voice_defender_noise_seed42.keras
    └── deep_voice_defender_full_seed42.keras
    ... (12 total)
```

---

## 7. Commands Reference (For Agents)

```bash
# Setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Baseline evaluation (Phase 1)
python src/evaluate.py

# Ablation training (Phase 2)
bash scripts/run_ablation.sh

# Cross-corpus test (Phase 3)
python src/test_cross_corpus.py

# Paper compile
cd paper && latexmk -pdf main.tex
```

---

## 8. Success Criteria (Definition of Done)

- [ ] **EER on ASVspoof LA eval** reported with 3-seed mean ± std for all 4 modes
- [ ] **Ablation table** shows statistically significant improvement from full augmentation
- [ ] **Cross-corpus results** show <20% relative EER degradation on WhatsApp audio
- [ ] **Paper** compiles to PDF, all tables/figures auto-generated from results/
- [ ] **README.md** updated with real numbers, no fake claims
- [ ] **GitHub repo** public, reproducible, includes all trained models

---

## 9. Notes for Future Agents

1. **Don't rewrite working code** — preprocessing, model, train, predict, app all work. Only add evaluation/ablation.
2. **ASVspoof protocol files** — train: `...cm.train.trn.txt`, eval: `...cm.eval.trl.txt`
3. **Audio paths** — train: `data/raw/LA/LA/ASVspoof2019_LA_train/flac/`, eval: `..._eval/flac/`
4. **Model input** — always (128, 128, 1) after `np.expand_dims(spec, 0)` and `np.expand_dims(..., -1)`
5. **Score interpretation** — sigmoid output: >0.5 = bonafide (human), <0.5 = spoof (fake)
6. **GPU memory** — batch_size=32 works on RTX 4060/5060 (8GB VRAM)

---

**Next Action:** Start Phase 1 — create `src/evaluate.py` and run baseline evaluation.